/* Shared shell: nav injection + API client. */
(function(){
"use strict";

const PAGES = [
  ["index.html",      "Launch"],
  ["console.html",    "Console"],
  ["system.html",     "System"],
  ["data.html",       "Data"],
  ["status.html",     "Status"]
];

function currentPage(){
  const f = location.pathname.split("/").pop();
  return (!f || f === "") ? "index.html" : f;
}

window.DRISHTIX_nav = function(theme){
  const here = currentPage();
  const links = PAGES.map(([href,label]) =>
    `<a href="${href}"${href===here?' aria-current="page"':''}>${label}</a>`
  ).join("");
  document.body.insertAdjacentHTML("afterbegin",
    `<nav class="nav">
       <span class="brand">DrishtiX<small>SIH26167 · ISRO</small></span>
       ${links}
       <span class="flag" aria-hidden="true"></span>
     </nav>`);
};

/* ── API client ───────────────────────────────────────────────────
   Every page degrades when the backend is not running: pages render,
   live panels say so plainly. A demo laptop with no server should
   still show the work. */
const API = {
  base: "",
  async health(){
    const r = await fetch(`${API.base}/api/health`);
    if(!r.ok) throw new Error(`health ${r.status}`);
    return r.json();
  },
  async models(){
    const r = await fetch(`${API.base}/api/models`);
    if(!r.ok) throw new Error(`models ${r.status}`);
    return r.json();
  },
  async scenes(){
    const r = await fetch(`${API.base}/api/scenes`);
    if(!r.ok) throw new Error(`scenes ${r.status}`);
    return r.json();
  },
  async ask(query, files){
    const fd = new FormData();
    fd.append("query", query);
    (files||[]).forEach(f => fd.append("images", f));
    const r = await fetch(`${API.base}/api/ask`, {method:"POST", body:fd});
    if(!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async askScene(query, scene, pair){
    const fd = new FormData();
    fd.append("query", query);
    fd.append("scene", scene);
    fd.append("pair", pair ? "true" : "false");
    const r = await fetch(`${API.base}/api/ask/scene`, {method:"POST", body:fd});
    if(!r.ok) throw new Error(await r.text());
    return r.json();
  }
};
window.DRISHTIX_api = API;

window.DRISHTIX_offline = function(el, err){
  el.innerHTML =
    `<p class="lede" style="font-size:14px">Backend not reachable. Start it with
     <code style="font-family:var(--mono)">uvicorn api.main:app --port 8000</code>
     and reload. Everything else on this page is static and still readable.</p>
     <p class="lede" style="font-size:12px;margin-top:8px;opacity:.7">${err||""}</p>`;
};
})();
