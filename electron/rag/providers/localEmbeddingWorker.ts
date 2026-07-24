// electron/rag/providers/localEmbeddingWorker.ts
//
// Worker-thread host for LocalEmbeddingProvider's ONNX inference. Mirrors the
// exact pattern used by electron/llm/intentClassifierWorker.ts.
//
// WHY (2026-07-05 SIGTRAP crash hardening): 9/9 real macOS crash reports
// showed the app crashing on the MAIN THREAD inside ONNX Runtime's BFC
// allocator during a live InferenceSession::Run() call — consistent with
// multiple ONNX sessions (Whisper STT worker + IntentClassifier worker +
// this local-embedding fallback) being concurrently active in-process. The
// embedding fallback was the only one of the three still running its
// pipeline()/inference DIRECTLY on the main process, so it is moved into its
// own worker_threads.Worker here, matching the isolation the other two
// already had. Also applies bounded intra/inter-op thread counts (see
// electron/utils/onnxThreadConfig.ts) to reduce native thread/memory
// pressure even when multiple sessions are concurrently active.
//
// Message protocol (mirrors intentClassifierWorker.ts):
//   { type: 'init', requestId, isPackaged, localModelPath, cacheDir }
//     -> { type: 'ready', requestId } | { type: 'error', requestId, error }
//   { type: 'embed', requestId, texts: string[] }
//     -> { type: 'result', requestId, vectors: number[][] } | { type: 'error', requestId, error }

import { parentPort } from 'worker_threads';
import { getBoundedOnnxSessionOptions } from '../../utils/onnxThreadConfig';
import { classifyWorkerFailure } from '../../utils/workerStatus';

if (!parentPort) throw new Error('localEmbeddingWorker must be run as a Worker thread');

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DIMENSIONS = 384;

let pipe: any = null;
let loadingPromise: Promise<void> | null = null;

// @huggingface/transformers is ESM-only — must use a true dynamic import().
// `new Function` keeps this opaque to TypeScript's commonjs rewrite (which
// would otherwise turn `import()` into `require()` and fail for an ESM-only
// package). See LocalEmbeddingProvider.ts for the full explanation.
async function loadTransformers(): Promise<{ pipeline: any; env: any }> {
  return (new Function('return import("@huggingface/transformers")')()) as any;
}

async function ensureLoaded(msg: any): Promise<void> {
  if (pipe) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    const { pipeline, env } = await loadTransformers();

    env.allowRemoteModels = false;
    env.localModelPath = msg.modelPath;

    console.log('[LocalEmbeddingWorker] Loading feature-extraction model (all-MiniLM-L6-v2)...');
    // dtype:'q8' loads model_quantized.onnx (the git-tracked variant) instead
    // of the fp32 model.onnx (untracked, downloaded via postinstall). A direct
    // q8-vs-fp32 eval showed identical retrieval ranking (5/5 top-1, top-3
    // overlap 3/3, self-cos >0.97), so this is a pure ~64MB size + install-speed
    // win with no retrieval-quality change. (MobileBERT stays fp32 — its q8
    // flips borderline intent labels, see intentClassifierWorker.ts.)
    pipe = await pipeline('feature-extraction', MODEL_ID, {
      local_files_only: true,
      dtype: 'q8',
      session_options: getBoundedOnnxSessionOptions(),
    });
    console.log('[LocalEmbeddingWorker] Feature-extraction model loaded successfully.');
    parentPort!.postMessage({ type: 'status', status: { type: 'ready', backend: 'onnx', modelPath: msg.modelPath } });
  })();

  try {
    await loadingPromise;
  } catch (e) {
    loadingPromise = null;
    pipe = null;
    const failure = classifyWorkerFailure(e);
    parentPort!.postMessage({
      type: 'status',
      status: {
        type: failure.recoverable ? 'degraded' : 'failed',
        backend: 'none',
        reason: failure.reason,
        message: failure.message,
        recoverable: failure.recoverable,
      },
    });
    throw e;
  }
}

parentPort.on('message', async (msg: any) => {
  try {
    if (msg.type === 'init') {
      await ensureLoaded(msg);
      parentPort!.postMessage({ type: 'ready', requestId: msg.requestId });
      return;
    }

    if (msg.type === 'embed') {
      if (!pipe) {
        await ensureLoaded(msg);
      }
      const texts: string[] = msg.texts;
      const output = await pipe(texts, { pooling: 'mean', normalize: true });
      const batchSize = texts.length;
      const vectors: number[][] = [];
      for (let i = 0; i < batchSize; i++) {
        vectors.push(Array.from(output.data.slice(i * DIMENSIONS, (i + 1) * DIMENSIONS)) as number[]);
      }
      parentPort!.postMessage({ type: 'result', requestId: msg.requestId, vectors });
      return;
    }

    parentPort!.postMessage({
      type: 'error',
      requestId: msg.requestId,
      error: `Unknown message type: ${msg.type}`,
    });
  } catch (e: any) {
    parentPort!.postMessage({
      type: 'error',
      requestId: msg.requestId,
      error: e?.message || String(e),
    });
  }
});
