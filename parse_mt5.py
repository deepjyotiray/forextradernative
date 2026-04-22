from bs4 import BeautifulSoup
import json

with open(r'C:\Users\deepj\Desktop\ReportHistory-106097742.html', encoding='utf-16') as f:
    soup = BeautifulSoup(f, 'html.parser')

rows = soup.find_all('tr')
trades = []
for row in rows:
    cells = [td.get_text(strip=True) for td in row.find_all('td')]
    hidden = [td.get_text(strip=True) for td in row.find_all('td', class_='hidden')]
    if len(cells) >= 12 and cells[2] == 'XAUUSD' and cells[3] in ('sell', 'buy'):
        if cells[8] and '2026' in cells[8]:
            strategy = hidden[0] if hidden else ''
            trades.append({
                'open_time': cells[0], 'position': cells[1], 'symbol': cells[2],
                'type': cells[3], 'volume': float(cells[4]),
                'open_price': float(cells[5]), 'sl': float(cells[6]) if cells[6] else 0,
                'tp': float(cells[7]) if cells[7] else 0,
                'close_time': cells[8], 'close_price': float(cells[9]),
                'commission': float(cells[10]), 'swap': float(cells[11]),
                'profit': float(cells[12]) if len(cells) > 12 else 0,
                'strategy': strategy
            })

total_pnl = sum(t['profit'] for t in trades)
wins = [t for t in trades if t['profit'] > 0]
losses = [t for t in trades if t['profit'] < 0]

print(json.dumps({
    'total_trades': len(trades),
    'wins': len(wins),
    'losses': len(losses),
    'total_pnl': round(total_pnl, 2),
    'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
    'avg_win': round(sum(t['profit'] for t in wins) / len(wins), 2) if wins else 0,
    'avg_loss': round(sum(t['profit'] for t in losses) / len(losses), 2) if losses else 0,
    'max_win': round(max(t['profit'] for t in wins), 2) if wins else 0,
    'max_loss': round(min(t['profit'] for t in losses), 2) if losses else 0,
    'trades': trades
}))
