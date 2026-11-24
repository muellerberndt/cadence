// One worker of page.html: runs its job and posts each line of output, then its end.
import { run } from './run.js';

self.onmessage = async ({ data }) => {
  const done = await run(data, (line) => self.postMessage({ line }));
  self.postMessage({ done });
};
