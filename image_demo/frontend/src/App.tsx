import {ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState} from 'react';
import {decode, RLEObject} from './utils/mask';
import './App.css';

type Segment = {
  id: number;
  bbox: number[];
  area: number;
  predicted_iou: number;
  stability_score: number;
  color: string;
  segmentation: {
    counts: string;
    size: [number, number];
  };
};

type SegmentResponse = {
  image: {
    width: number;
    height: number;
  };
  count: number;
  segments: Segment[];
  saved_to?: string;
  model_size?: string;
};

const defaultEndpoint = (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  'http://localhost:5050';
const modelOptions = [
  {value: 'tiny', label: 'Tiny'},
  {value: 'small', label: 'Small'},
  {value: 'base_plus', label: 'Base Plus'},
  {value: 'large', label: 'Large'},
];

export default function App() {
  const [backendUrl, setBackendUrl] = useState(defaultEndpoint);
  const [maxMasks, setMaxMasks] = useState(30);
  const [saveOnServer, setSaveOnServer] = useState(true);
  const [modelSize, setModelSize] = useState('base_plus');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [responseMetadata, setResponseMetadata] = useState<SegmentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  const sanitizedEndpoint = useMemo(() => backendUrl.replace(/\/$/, ''), [backendUrl]);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setImageFile(null);
      setImageUrl(null);
      setSegments([]);
      setResponseMetadata(null);
      return;
    }

    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setSegments([]);
    setResponseMetadata(null);
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!imageFile) {
      setError('Select an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('max_masks', String(maxMasks));
    formData.append('model_size', modelSize);
    if (saveOnServer) {
      formData.append('save', '1');
    }

    try {
      const response = await fetch(`${sanitizedEndpoint}/segment-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Segmentation request failed');
      }

      const data = (await response.json()) as SegmentResponse;
      setSegments(data.segments ?? []);
      setResponseMetadata(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Unknown error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) {
      return;
    }

    if (!image.complete) {
      image.onload = () => drawMasks(canvas, image, segments);
      return;
    }

    drawMasks(canvas, image, segments);
  }, [segments, imageUrl]);

  const downloadJson = () => {
    if (!responseMetadata) {
      return;
    }
    const blob = new Blob([JSON.stringify(responseMetadata, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'segment-anything.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      <header>
        <div>
          <h1>SAM 2 Image Segmenter</h1>
          <p>Upload an image, generate mask proposals, and export JSON without GraphQL.</p>
        </div>
      </header>

      <main>
        <section className="panel">
          <form className="controls" onSubmit={onSubmit}>
            <label className="field">
              <span>Backend URL</span>
              <input
                type="text"
                value={backendUrl}
                onChange={event => setBackendUrl(event.target.value)}
              />
            </label>

            <label className="field">
              <span>Choose Image</span>
              <input type="file" accept="image/*" onChange={onFileChange} />
            </label>

            <label className="field">
              <span>Max Masks</span>
              <input
                type="number"
                min={1}
                max={200}
                value={maxMasks}
                onChange={event => setMaxMasks(parseInt(event.target.value || '1', 10))}
              />
            </label>

            <label className="field">
              <span>Model</span>
              <select value={modelSize} onChange={event => setModelSize(event.target.value)}>
                {modelOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={saveOnServer}
                onChange={event => setSaveOnServer(event.target.checked)}
              />
              Save JSON on backend
            </label>

            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Segmenting…' : 'Segment Image'}
            </button>
          </form>

          {error && <p className="error">{error}</p>}
          {responseMetadata?.saved_to && (
            <p className="hint">Saved on server: {responseMetadata.saved_to}</p>
          )}
        </section>

        <section className="viewer-panel">
          {imageUrl ? (
            <div className="viewer">
              <img ref={imageRef} src={imageUrl} alt="uploaded" />
              <canvas ref={canvasRef} />
            </div>
          ) : (
            <div className="placeholder">Upload an image to see the preview.</div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Segments ({segments.length})</h2>
            <div className="panel-actions">
              <button onClick={downloadJson} disabled={!responseMetadata}>
                Download JSON
              </button>
            </div>
          </div>

          {responseMetadata?.model_size && (
            <p className="hint">Generated with model: {responseMetadata.model_size}</p>
          )}

          {segments.length === 0 ? (
            <p className="hint">Run segmentation to populate mask proposals.</p>
          ) : (
            <div className="segment-grid">
              {segments.map(segment => (
                <article key={segment.id} className="segment-card">
                  <div className="swatch" style={{backgroundColor: segment.color}} />
                  <div>
                    <p className="segment-title">Mask #{segment.id}</p>
                    <p className="segment-meta">
                      area: {segment.area.toFixed(0)} px² • IoU:{' '}
                      {segment.predicted_iou.toFixed(3)} • stability:{' '}
                      {segment.stability_score.toFixed(3)}
                    </p>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function drawMasks(canvas: HTMLCanvasElement, image: HTMLImageElement, segments: Segment[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  const width = image.naturalWidth;
  const height = image.naturalHeight;
  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, width, height);
  if (segments.length === 0) {
    return;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.globalAlpha = 1;

  segments.forEach(segment => {
    const rle: RLEObject = {
      counts: segment.segmentation.counts,
      size: segment.segmentation.size,
    };

    const decoded = decode([rle]);
    const maskData = decoded.data as Uint8Array;
    const maskHeight = decoded.shape[0];
    const maskWidth = decoded.shape[1];

    const [r, g, b] = hexToRgb(segment.color);
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = maskWidth;
    maskCanvas.height = maskHeight;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) {
      return;
    }

    const maskImage = maskCtx.createImageData(maskWidth, maskHeight);
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] === 0) {
        continue;
      }
      const offset = i * 4;
      maskImage.data[offset] = r;
      maskImage.data[offset + 1] = g;
      maskImage.data[offset + 2] = b;
      maskImage.data[offset + 3] = 140;
    }

    maskCtx.putImageData(maskImage, 0, 0);
    ctx.drawImage(maskCanvas, 0, 0, maskWidth, maskHeight, 0, 0, width, height);
  });
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace('#', '');
  const bigint = parseInt(value, 16);
  return [
    (bigint >> 16) & 255,
    (bigint >> 8) & 255,
    bigint & 255,
  ];
}
