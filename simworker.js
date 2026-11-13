/* Web Worker : calcule les simulations du Swiss hors du thread principal et renvoie des resultats progressifs. */
importScripts('sim.js');
onmessage = e => {
  const { prep, n } = e.data;
  const res = SIM.runPrepared(prep, n, partial => postMessage({ res: partial, done: false }));
  postMessage({ res, done: true });
};
