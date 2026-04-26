"""
Improved Backtest Job Manager with Balanced Chunk Distribution
Fixes non-uniform progress by creating chunks based on tick density rather than time periods.
"""
import os
import json
import math
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from .backtest_jobs import BacktestJobManager, BacktestJob, _run_chunk_worker
from .progress_optimizer import AdaptiveChunkManager, ProgressMonitor, optimize_chunk_distribution


class BalancedBacktestJobManager(BacktestJobManager):
    """Enhanced job manager with balanced chunk distribution."""
    
    def __init__(self):
        super().__init__()
        self.chunk_manager = AdaptiveChunkManager()
        self.progress_monitor = ProgressMonitor()
        
    def _build_balanced_split_ranges(self, start_utc: datetime, end_utc: datetime, 
                                   split_mode: str, parallel_workers: int) -> List[Tuple[datetime, datetime]]:
        """Build balanced split ranges based on estimated tick density."""
        
        print(f"Creating balanced chunks for {split_mode} mode with {parallel_workers} workers...")
        
        if split_mode == "BALANCED_TICK_DENSITY":
            # Use our new balanced approach
            return optimize_chunk_distribution(start_utc, end_utc, parallel_workers)
        
        elif split_mode in ["DAILY", "WEEKLY", "MONTHLY", "QUARTERLY"]:
            # Use original time-based splitting but with balancing
            original_ranges = self._build_time_based_ranges(start_utc, end_utc, split_mode)
            return self._balance_existing_ranges(original_ranges, parallel_workers)
        
        else:
            # Fallback to original method
            return super()._build_split_ranges(start_utc, end_utc, split_mode)
    
    def _build_time_based_ranges(self, start_utc: datetime, end_utc: datetime, 
                               split_mode: str) -> List[Tuple[datetime, datetime]]:
        """Build time-based ranges (original logic)."""
        
        ranges = []
        current = start_utc
        
        while current < end_utc:
            if split_mode == "DAILY":
                next_boundary = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif split_mode == "WEEKLY":
                days_to_monday = 7 - current.weekday()
                next_boundary = (current + timedelta(days=days_to_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif split_mode == "MONTHLY":
                if current.month == 12:
                    next_boundary = current.replace(year=current.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                else:
                    next_boundary = current.replace(month=current.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            elif split_mode == "QUARTERLY":
                quarter_start_month = ((current.month - 1) // 3) * 3 + 1
                next_q_month = quarter_start_month + 3
                year = current.year
                if next_q_month > 12:
                    next_q_month -= 12
                    year += 1
                next_boundary = current.replace(year=year, month=next_q_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_boundary = current + timedelta(days=1)
            
            end_time = min(next_boundary, end_utc)
            ranges.append((current, end_time))
            current = next_boundary
        
        return ranges
    
    def _balance_existing_ranges(self, ranges: List[Tuple[datetime, datetime]], 
                               parallel_workers: int) -> List[Tuple[datetime, datetime]]:
        """Balance existing time-based ranges by merging small chunks and splitting large ones."""
        
        print(f"Balancing {len(ranges)} time-based ranges...")
        
        # Estimate tick density for each range
        range_info = []
        for start, end in ranges:
            estimated_ticks = self.chunk_manager.tick_density_estimator._estimate_tick_density(start, end)
            duration_hours = (end - start).total_seconds() / 3600
            
            range_info.append({
                'start': start,
                'end': end,
                'estimated_ticks': estimated_ticks,
                'duration_hours': duration_hours,
                'ticks_per_hour': estimated_ticks / duration_hours if duration_hours > 0 else 0
            })
        
        # Calculate target ticks per chunk
        total_ticks = sum(r['estimated_ticks'] for r in range_info)
        target_chunks = max(parallel_workers, min(parallel_workers * 4, len(ranges)))
        target_ticks_per_chunk = total_ticks / target_chunks
        
        print(f"Target: {target_ticks_per_chunk:,.0f} ticks per chunk ({target_chunks} chunks)")
        
        # Balance the ranges
        balanced_ranges = []
        current_chunk_ticks = 0
        current_chunk_start = None
        
        for range_data in range_info:
            if current_chunk_start is None:
                current_chunk_start = range_data['start']
            
            current_chunk_ticks += range_data['estimated_ticks']
            
            # Check if we should close this chunk
            should_close = (
                current_chunk_ticks >= target_ticks_per_chunk * 0.8 or  # Reached target
                range_data == range_info[-1]  # Last range
            )
            
            if should_close:
                balanced_ranges.append((current_chunk_start, range_data['end']))
                
                print(f"  Chunk {len(balanced_ranges)}: "
                      f"{current_chunk_start.strftime('%m-%d %H:%M')} -> {range_data['end'].strftime('%m-%d %H:%M')} "
                      f"({current_chunk_ticks:,.0f} ticks)")
                
                current_chunk_start = None
                current_chunk_ticks = 0
        
        print(f"Balanced to {len(balanced_ranges)} chunks")
        return balanced_ranges
    
    def _run_job_parallel_chunks_balanced(self, job: BacktestJob) -> Dict:
        """Run job with balanced parallel chunks."""
        
        req = job.request
        
        # Use balanced chunk creation
        if hasattr(req, 'split_mode') and req.split_mode in ["DAILY", "WEEKLY", "MONTHLY", "QUARTERLY"]:
            ranges = self._build_balanced_split_ranges(
                req.start_utc, req.end_utc, req.split_mode, req.parallel_workers
            )
        else:
            # Use tick-density based balancing for best results
            ranges = optimize_chunk_distribution(
                req.start_utc, req.end_utc, req.parallel_workers
            )
        
        if not ranges:
            raise RuntimeError("No balanced ranges generated")
        
        # Initialize progress monitoring
        job.total_ticks = len(ranges)  # Use chunk count as progress unit
        job.processed_ticks = 0
        job.chunk_states = []
        
        for i, (start, end) in enumerate(ranges):
            estimated_ticks = self.chunk_manager.tick_density_estimator._estimate_tick_density(start, end)
            
            chunk_state = {
                'index': i,
                'status': 'queued',
                'progress_pct': 0.0,
                'sim_time_utc': None,
                'start_utc': start.isoformat(),
                'end_utc': end.isoformat(),
                'estimated_ticks': estimated_ticks,
                'start_timestamp': time.time()
            }
            job.chunk_states.append(chunk_state)
            self.progress_monitor.start_chunk_monitoring(i)
        
        # Run parallel processing with enhanced monitoring
        return self._execute_balanced_parallel_chunks(job, ranges)
    
    def _execute_balanced_parallel_chunks(self, job: BacktestJob, 
                                        ranges: List[Tuple[datetime, datetime]]) -> Dict:
        """Execute balanced parallel chunks with real-time monitoring."""
        
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        req = job.request
        max_workers = min(req.parallel_workers, len(ranges))
        
        print(f"Executing {len(ranges)} balanced chunks with {max_workers} workers...")
        
        # Enhanced progress tracking
        chunk_results = []
        completed_chunks = 0
        
        with mp.Manager() as mgr:
            progress_queue = mgr.Queue()
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks
                futures_map = {}
                
                for idx, (chunk_start, chunk_end) in enumerate(ranges):
                    payload = {
                        'symbol': req.symbol,
                        'strategy': req.strategy,
                        'start_utc': chunk_start.isoformat(),
                        'end_utc': chunk_end.isoformat(),
                        'initial_balance': req.initial_balance,
                        'use_runtime_config': req.use_runtime_config,
                        'max_ticks_per_batch': req.max_ticks_per_batch,
                        'replay_max_points': req.replay_max_points,
                        'fast_mode': req.fast_mode,
                        'split_mode': 'NONE',
                        'parallel_workers': 1,
                        'cache_only': req.cache_only,
                        'chunk_index': idx,
                        'progress_queue': progress_queue
                    }
                    
                    job.chunk_states[idx]['status'] = 'running'
                    future = executor.submit(_run_chunk_worker, payload)
                    futures_map[future] = (idx, chunk_start, chunk_end)
                
                # Monitor progress with enhanced tracking
                pending_futures = set(futures_map.keys())
                last_progress_update = time.time()
                
                while pending_futures:
                    if job.is_cancelled():
                        self._append_log(job, "Cancellation requested")
                        for f in pending_futures:
                            f.cancel()
                        raise RuntimeError("Job cancelled")
                    
                    # Process progress updates
                    self._process_progress_updates(job, progress_queue)
                    
                    # Check for completed futures
                    done_futures = [f for f in pending_futures if f.done()]
                    
                    for future in done_futures:
                        pending_futures.remove(future)
                        idx, chunk_start, chunk_end = futures_map[future]
                        
                        try:
                            result = future.result()
                            
                            if not result.get('ok'):
                                raise RuntimeError(f"Chunk {idx} failed: {result.get('error')}")
                            
                            chunk_results.append({
                                'index': idx,
                                'start_utc': chunk_start,
                                'end_utc': chunk_end,
                                'result': result['result']
                            })
                            
                            # Update chunk state
                            job.chunk_states[idx]['status'] = 'completed'
                            job.chunk_states[idx]['progress_pct'] = 100.0
                            job.chunk_states[idx]['sim_time_utc'] = chunk_end.isoformat()
                            
                            completed_chunks += 1
                            job.processed_ticks = completed_chunks
                            job.progress_pct = (completed_chunks / len(ranges)) * 100.0
                            
                            self._append_log(job, f"Chunk {idx+1}/{len(ranges)} completed "
                                           f"({chunk_start.strftime('%m-%d')} -> {chunk_end.strftime('%m-%d')})")
                            
                        except Exception as e:
                            job.chunk_states[idx]['status'] = 'failed'
                            self._append_log(job, f"Chunk {idx} failed: {e}")
                            raise RuntimeError(f"Chunk {idx} execution failed: {e}")
                    
                    # Periodic progress update
                    now = time.time()
                    if now - last_progress_update >= 2.0:  # Every 2 seconds
                        self._update_job_progress_summary(job)
                        last_progress_update = now
                    
                    if pending_futures:
                        time.sleep(0.1)  # Short sleep to prevent busy waiting
        
        # Sort results by index
        chunk_results.sort(key=lambda x: x['index'])
        
        # Generate progress analysis report
        progress_analysis = self.progress_monitor.get_progress_analysis()
        self._append_log(job, f"Progress analysis: {json.dumps(progress_analysis, indent=2)}")
        
        # Aggregate results
        return self._aggregate_balanced_chunk_results(req, chunk_results, max_workers)
    
    def _process_progress_updates(self, job: BacktestJob, progress_queue):
        """Process progress updates from worker processes."""
        
        updates_processed = 0
        
        while updates_processed < 100:  # Limit to prevent blocking
            try:
                update = progress_queue.get_nowait()
                chunk_idx = update.get('chunk_index', -1)
                
                if 0 <= chunk_idx < len(job.chunk_states):
                    processed_ticks = update.get('processed_ticks', 0)
                    total_ticks = update.get('total_ticks', 1)
                    progress_pct = (processed_ticks / total_ticks) * 100.0 if total_ticks > 0 else 0
                    
                    job.chunk_states[chunk_idx]['progress_pct'] = round(progress_pct, 2)
                    job.chunk_states[chunk_idx]['sim_time_utc'] = update.get('sim_time_utc')
                    
                    # Update progress monitor
                    self.progress_monitor.update_chunk_progress(
                        chunk_idx, progress_pct, processed_ticks, 
                        update.get('sim_time_utc', '')
                    )
                
                updates_processed += 1
                
            except:
                break  # No more updates available
    
    def _update_job_progress_summary(self, job: BacktestJob):
        """Update overall job progress summary."""
        
        if not job.chunk_states:
            return
        
        # Calculate overall progress
        total_progress = sum(chunk['progress_pct'] for chunk in job.chunk_states)
        job.progress_pct = total_progress / len(job.chunk_states)
        
        # Update simulation time to latest
        sim_times = [chunk.get('sim_time_utc') for chunk in job.chunk_states 
                    if chunk.get('sim_time_utc')]
        if sim_times:
            job.sim_time_utc = max(sim_times)
        
        # Save state
        with self._lock:
            self._save_state_locked()
    
    def _aggregate_balanced_chunk_results(self, req, chunk_results: List[Dict], 
                                        workers_used: int) -> Dict:
        """Aggregate results from balanced chunks."""
        
        print(f"Aggregating {len(chunk_results)} balanced chunk results...")
        
        # Use the original aggregation logic but with enhanced metadata
        result = super()._aggregate_chunk_outputs(req, chunk_results, workers_used)
        
        # Add balancing metadata
        if 'meta' not in result:
            result['meta'] = {}
        
        result['meta'].update({
            'chunk_balancing': {
                'balancing_enabled': True,
                'chunk_count': len(chunk_results),
                'balancing_method': 'tick_density_based',
                'workers_used': workers_used
            }
        })
        
        # Add chunk performance analysis
        chunk_performance = []
        for chunk_result in chunk_results:
            chunk_data = chunk_result['result']
            chunk_summary = chunk_data.get('summary', {})
            
            chunk_performance.append({
                'index': chunk_result['index'],
                'start_utc': chunk_result['start_utc'].isoformat(),
                'end_utc': chunk_result['end_utc'].isoformat(),
                'trades': chunk_summary.get('trades', 0),
                'pnl': chunk_summary.get('pnl', 0),
                'ticks_processed': chunk_data.get('meta', {}).get('ticks_replayed', 0)
            })
        
        result['meta']['chunk_performance'] = chunk_performance
        
        return result
    
    def create_job(self, req) -> BacktestJob:
        """Create job with enhanced balancing support."""
        
        # Add balanced split mode if not specified
        if not hasattr(req, 'split_mode') or req.split_mode == 'NONE':
            # Auto-select balanced mode for parallel jobs
            if hasattr(req, 'parallel_workers') and req.parallel_workers > 1:
                req.split_mode = 'BALANCED_TICK_DENSITY'
        
        return super().create_job(req)
    
    def _run_job(self, job_id: str):
        """Enhanced job runner with balanced processing."""
        
        job = self.get_job(job_id)
        if not job:
            return
        
        # Check if we should use balanced parallel processing
        if (hasattr(job.request, 'split_mode') and 
            job.request.split_mode in ['BALANCED_TICK_DENSITY', 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY']):
            
            try:
                job.status = 'running'
                job.started_at = datetime.now(timezone.utc).isoformat()
                
                self._append_log(job, "Starting balanced parallel execution")
                
                result = self._run_job_parallel_chunks_balanced(job)
                
                if job.is_cancelled():
                    job.status = 'cancelled'
                    return
                
                # Complete job
                from .backtest_sim import persist_backtest_artifacts
                artifacts = persist_backtest_artifacts(job.job_id, result, 
                                                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtests"))
                
                job.result = result
                job.artifacts = artifacts
                job.status = 'completed'
                job.progress_pct = 100.0
                job.ended_at = datetime.now(timezone.utc).isoformat()
                
                self._append_log(job, "Balanced parallel execution completed successfully")
                
                with self._lock:
                    self._save_state_locked()
                
            except Exception as e:
                job.status = 'failed'
                job.error = str(e)
                job.ended_at = datetime.now(timezone.utc).isoformat()
                self._append_log(job, f"Balanced execution failed: {e}")
                
                with self._lock:
                    self._save_state_locked()
        else:
            # Use original job runner
            super()._run_job(job_id)


# Replace the global backtest_jobs instance
balanced_backtest_jobs = BalancedBacktestJobManager()