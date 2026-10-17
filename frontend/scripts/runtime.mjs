// Keep installed packages out of cloud-synced Desktop folders. No app secrets
// or source files are copied into the runtime cache.
import {createHash} from 'node:crypto';
import {existsSync,lstatSync,mkdirSync,mkdtempSync,readFileSync,realpathSync,renameSync,rmSync,symlinkSync,writeFileSync} from 'node:fs';
import {homedir} from 'node:os';
import {delimiter,dirname,join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
// npm and package shebangs must use the same Node selected by the caller.
process.env.PATH=dirname(process.execPath)+delimiter+(process.env.PATH??'');
const lock=readFileSync(join(root,'package-lock.json'));
const key=createHash('sha256').update(lock).update(`${process.platform}-${process.arch}-${process.versions.node.split('.')[0]}`).digest('hex').slice(0,16);
const base=process.env.TACTIQAI_FRONTEND_RUNTIME_ROOT??(process.platform==='darwin'?join(homedir(),'Library','Caches','TactiqAI','frontend'):join(homedir(),'.cache','tactiqai','frontend'));
const runtime=join(base,key);
if(!existsSync(join(runtime,'.ready'))){
 mkdirSync(base,{recursive:true});
 const staging=mkdtempSync(join(base,'.install-'));
 writeFileSync(join(staging,'package.json'),readFileSync(join(root,'package.json')));
 writeFileSync(join(staging,'package-lock.json'),lock);
 console.log('Preparing frontend dependencies outside cloud storage (first run for this lockfile)…');
 const install=spawnSync(process.platform==='win32'?'npm.cmd':'npm',['ci','--prefer-offline','--no-audit','--no-fund'],{cwd:staging,stdio:'inherit'});
 if(install.status!==0){rmSync(staging,{recursive:true,force:true});process.exit(install.status??1)}
 writeFileSync(join(staging,'.ready'),key);
 try{renameSync(staging,runtime)}catch(error){
  if(!existsSync(join(runtime,'.ready')))throw error;
  rmSync(staging,{recursive:true,force:true});
 }
}
const modules=join(root,'node_modules'),target=join(runtime,'node_modules');
let current;
try{current=lstatSync(modules)}catch(error){if(error.code!=='ENOENT')throw error}
let linked=false;
try{linked=Boolean(current?.isSymbolicLink()&&realpathSync(modules)===realpathSync(target))}catch(error){if(error.code!=='ENOENT')throw error}
if(!linked){
 if(current){
  if(current.isSymbolicLink())rmSync(modules);
  else{
   const backup=modules+'.cloud-backup-'+Date.now();renameSync(modules,backup);
   console.log(`Previous dependencies preserved at ${backup}`);
  }
 }
 symlinkSync(target,modules,process.platform==='win32'?'junction':'dir');
}
// Each lockfile/platform gets its own optimizer cache on the local disk.
process.env.TACTIQAI_VITE_CACHE_DIR=join(runtime,'vite-cache');
const [command='dev',...args]=process.argv.slice(2);
function run(binary,options){
 const child=spawnSync(join(target,'.bin',binary+(process.platform==='win32'?'.cmd':'')),options,{cwd:root,stdio:'inherit',env:process.env});
 if(child.error)throw child.error;
 if(child.status!==0)process.exit(child.status??1);
}
if(command==='prepare')console.log(`Frontend runtime ready: ${runtime}`);
else if(command==='dev')run('vite',['--host','127.0.0.1',...args]);
else if(command==='build'){run('tsc',['-b']);run('vite',['build',...args])}
else if(command==='test')run('vitest',['run',...args]);
else if(command==='preview')run('vite',['preview','--host','127.0.0.1',...args]);
else throw new Error(`Unknown frontend command: ${command}`);
