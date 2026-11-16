// Load Pyodide and NumPy, unpack the bundle (the library's source, the kernel, these scripts)
// at /w, run one script with its arguments; each line of output goes to `say`.
export async function run(job, say) {
  try {
    const base = job.pyodide || '/pyodide/';
    const { loadPyodide } = await import(base + 'pyodide.mjs');
    const py = await loadPyodide({ indexURL: base, stdout: say, stderr: say });
    await py.loadPackage('numpy', { messageCallback: () => {} });
    const bundle = await (await fetch('/bundle.tar', { cache: 'no-store' })).arrayBuffer();
    py.FS.mkdirTree('/w');
    py.unpackArchive(bundle, 'tar', { extractDir: '/w' });
    py.globals.set('ARGS', py.toPy([job.script, ...job.args]));
    py.runPython("import sys, os; sys.argv = list(ARGS); os.chdir('/w'); sys.path.insert(0, '/w/browser')");
    const code = py.FS.readFile('/w/' + job.script, { encoding: 'utf8' });
    await py.runPythonAsync(code, { filename: '/w/' + job.script });
    return 'ok';
  } catch (error) {
    say(String((error && error.message) || error).slice(-4000));
    return 'error';
  }
}
