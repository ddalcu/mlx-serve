// English-key completeness and pre-paint boot; hermetic, no npm or server needed.
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {test} from 'node:test';
const read = name => readFileSync(new URL('../src/html/'+name, import.meta.url), 'utf8');

// A lexical walk skips comments and template text, but enters ${expressions}.
// Only explicit t/th/N calls and data-i18n attributes are translation contracts.
export function markedKeys(source) {
  const keys = new Set();
  let i = 0;
  const markup = text => {
    for (const m of text.matchAll(/\bdata-i18n(?:-title|-aria-label|-placeholder|-alt)?="([^"]*)"/g))
      keys.add(m[1].replaceAll('&quot;', '"').replaceAll('&#39;', "'").replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&'));
  };
  function quoted() {
    const start = i, q = source[i++];
    while (i < source.length) { const c=source[i++]; if(c==='\\')i++; else if(c===q)break; }
    return vm.runInNewContext(source.slice(start,i));
  }
  function code(nested = false) {
    let depth = 0;
    while(i < source.length) {
      if(source.startsWith('//',i)) {i=source.indexOf('\n',i);if(i<0){i=source.length;return;}continue;}
      if(source.startsWith('/*',i)) {const end=source.indexOf('*/',i+2);i=end<0?source.length:end+2;continue;}
      const c=source[i];
      if(c==='/' && /(?:[=(:,!&|?;{}\[>]|\breturn)\s*$/.test(source.slice(Math.max(0,i-30),i))) {
        i++;let bracket=false;
        while(i<source.length) {const ch=source[i++];if(ch==='\\'){i++;continue;}if(ch==='[')bracket=true;if(ch===']')bracket=false;if(ch==='/'&&!bracket)break;}
        while(/[a-z]/i.test(source[i]||' '))i++;
        continue;
      }
      if(c==='"'||c==="'") {markup(quoted());continue;}
      if(c==='`') {
        i++;let raw='';
        while(i<source.length) {
          if(source[i]==='\\'){raw+=source.slice(i,i+2);i+=2;continue;}
          if(source[i]==='`'){i++;markup(raw);break;}
          if(source.startsWith('${',i)){markup(raw);raw='';i+=2;code(true);continue;}
          raw+=source[i++];
        }
        continue;
      }
      if(c==='{' )depth++;
      if(c==='}') {if(nested && depth===0){i++;return;}depth--;}
      const call = /^(?:t|th|N)\s*\(\s*/.exec(source.slice(i));
      if(call && !/[\w$.]/.test(source[i-1]||' ')) {
        i+=call[0].length;
        if(source[i]==='"'||source[i]==="'")keys.add(quoted());
        continue;
      }
      i++;
    }
  }
  code();return keys;
}
function boot(override, languages, blocked = false) {
  const root={lang:'',dataset:{},hasAttribute:()=>true};
  const context=vm.createContext({URL,EventTarget,AbortController,setTimeout,clearTimeout,
    navigator:{languages,language:languages[0]||'en'},
    localStorage:{getItem(){if(blocked)throw Error('blocked');return override;},setItem(){}},
    document:{documentElement:root,addEventListener(){},querySelectorAll(){return[];}},
    window:{addEventListener(){}}});
  vm.runInContext(read('api.js')+'\n'+read('i18n.js')+'\n;globalThis.I=studioModules["src/ui/i18n.js"];',context);
  return {root,I:context.I};
}
test('the language boot respects stored choice, browser zh variants and blocked storage',()=>{
  assert.equal(boot(null,['zh-CN']).root.lang,'zh-Hans');
  assert.equal(boot('en',['zh-TW']).root.lang,'en');
  assert.equal(boot('zh-Hans',['en']).root.lang,'zh-Hans');
  assert.equal(boot('system',['en','zh-SG']).root.lang,'zh-Hans');
  assert.equal(boot('bad',['fr']).root.lang,'en');
  assert.equal(boot('en',['zh-CN'],true).root.lang,'zh-Hans');
});
test('placeholders and English fallback preserve literal values',()=>{
  const {I}=boot('zh-Hans',['en']);
  assert.equal(I.t('Models'),'模型');
  assert.equal(I.t('unknown %@ / %@',['$&','<b>']),'unknown $& / <b>');
  assert.equal(I.t('unknown %@ %@',['one']),'unknown one %@');
});
test('every marked markup and JavaScript key has a Simplified Chinese entry',()=>{
  const {I}=boot('en',['en']);
  const keys=new Set(['Skip to content']);
  for(const name of ['api.js','app.js']) {
    // Vendored libraries and the dictionary are not application call sites.
    const modules = read(name).split(/(?=^\/\/ src\/[^\n]+\.m?js\nstudioModules\[)/m);
    for(const module of modules) if(/^\/\/ src\/(?:ui|core)\//.test(module) && !/^\/\/ src\/ui\/(?:zh-hans)\.js/.test(module))
      for(const key of markedKeys(module))keys.add(key);
  }
  for(const key of markedKeys('`'+read('index.html')+'`'))keys.add(key);
  const missing=[...keys].filter(k=>!Object.hasOwn(I.ZH,k)||!I.ZH[k]);
  assert.deepEqual(missing,[],'missing zh-Hans translations');
  for(const key of keys)assert.equal((I.ZH[key].match(/%@/g)||[]).length,(key.match(/%@/g)||[]).length,key+' placeholders');
  console.log(`${keys.size} marked keys covered`);
});
test('the key collector ignores comments and enters nested template expressions',()=>{
  assert.deepEqual([...markedKeys('// t("comment")\n/* N("comment2") */\n`text ${t("real")} ${`nested ${th("also real")}`}<b data-i18n="Markup">`')],['real','also real','Markup']);
});

test('language boot precedes CSS and English API documentation survives without JavaScript',()=>{
  const page=read('index.html');
  assert.ok(page.indexOf('<script>') < page.indexOf('<style>'));
  assert.match(read('i18n.js'), /bootLanguage/);
  assert.ok(page.includes('<noscript><h2>API reference</h2>'));
  assert.ok(page.includes('<th>Description</th>'));
});
