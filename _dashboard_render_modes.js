const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('dashboard.html','utf8');
const script = (html.match(/<script>([\s\S]*?)<\/script>/)||[])[1] || '';
const status = JSON.parse(fs.readFileSync('status-sample.json','utf8'));
const config = JSON.parse(fs.readFileSync('config-sample.json','utf8'));
function makeEl(id){
  return {
    id,
    innerHTML:'', textContent:'', className:'', checked:false, disabled:false, value:'', href:'', title:'',
    style:{}, dataset:{},
    addEventListener(){}, removeEventListener(){}, appendChild(){}, remove(){}, focus(){}, click(){},
    contains(){ return false; },
    setAttribute(k,v){ this[k]=v; }, getAttribute(k){ return this[k] || ''; },
    getBoundingClientRect(){ return {top:0,left:0,width:0,height:0}; },
    scrollTop:0, clientHeight:100, scrollHeight:100,
  };
}
const elements = new Map();
const document = {
  hidden:false,
  title:'',
  body: makeEl('body'),
  documentElement: makeEl('html'),
  createElement(tag){ return makeEl(tag); },
  getElementById(id){ if(!elements.has(id)) elements.set(id, makeEl(id)); return elements.get(id); },
  querySelector(){ return makeEl('q'); },
  querySelectorAll(){ return []; },
  addEventListener(){},
};
function makeContext(){
  const location = { origin:'http://127.0.0.1:8000', protocol:'http:', host:'127.0.0.1:8000' };
  const fakeFetch = async (url)=>({ ok:true, json: async()=> url.includes('/config') ? config : status });
  function WS(){ this.close=()=>{}; }
  const context = {
    console, document, location,
    window:null,
    navigator:{ userAgent:'node' },
    localStorage:{ getItem(){return null;}, setItem(){}, removeItem(){} },
    sessionStorage:{ getItem(){return null;}, setItem(){}, removeItem(){} },
    fetch: fakeFetch,
    WebSocket: WS,
    alert: (...args)=>console.log('alert', ...args),
    open: ()=>{},
    setTimeout: (fn)=>0,
    clearTimeout: ()=>{},
    setInterval: (fn)=>0,
    clearInterval: ()=>{},
    requestAnimationFrame: (fn)=>{ fn(); return 0; },
    cancelAnimationFrame: ()=>{},
    Date, Math, JSON, URL, URLSearchParams,
  };
  context.window = context;
  context.window.innerWidth = 1440;
  context.window.getSelection = ()=>({ rangeCount:0, isCollapsed:true, anchorNode:null });
  return context;
}
function testPayload(label, payload){
  const context = makeContext();
  vm.createContext(context);
  try {
    vm.runInContext(script, context, { timeout: 5000 });
    context.render(payload, { full:true });
    console.log(label + ': render ok');
  } catch (e) {
    console.error(label + ': render failed');
    console.error(e && e.stack || e);
  }
}
testPayload('status', status);
testPayload('config', config);
testPayload('merged', Object.assign({}, status, config));
