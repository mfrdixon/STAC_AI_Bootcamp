import {run} from '../public/decision-lab/engine.mjs';
let text='';for await(const chunk of process.stdin)text+=chunk;
try{console.log(JSON.stringify(run(JSON.parse(text||'{}'))));}catch(e){console.error(e.message);process.exitCode=1;}
