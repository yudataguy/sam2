import {ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {decode, RLEObject} from './utils/mask';
import './App.css';

type Segment = {
  id: number | string;
  bbox: number[];
  area: number;
  predicted_iou: number;
  stability_score: number;
  color: string;
  segmentation: {
    counts: string;
    size: [number, number];
  };
  label?: string;
  refined_from?: number | string;
  remainder_of?: number | string;
  parent_id?: number | string; // For hierarchical IDs (parent of 1.1 is 1)
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

  // Tab state
  const [activeTab, setActiveTab] = useState<'segments' | 'refine'>('segments');

  // Refinement state
  const [isDrawing, setIsDrawing] = useState(false);
  const [bboxStart, setBboxStart] = useState<{x: number; y: number} | null>(null);
  const [bboxCurrent, setBboxCurrent] = useState<{x: number; y: number} | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<number | string | null>(null);
  const [refinementMode, setRefinementMode] = useState<'refine' | 'create'>('refine'); // 'refine' or 'create'
  const [highlightedSegmentId, setHighlightedSegmentId] = useState<number | string | null>(null);

  // Labeling state
  const [labels, setLabels] = useState<string[]>(() => {
    const saved = localStorage.getItem('sam2_labels');
    return saved ? JSON.parse(saved) : ['apple', 'orange', 'strawberry'];
  });
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const [newLabelInput, setNewLabelInput] = useState('');
  const [showLabelSettings, setShowLabelSettings] = useState(false);

  // Save state
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const autoSaveTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  // Persist labels to localStorage
  useEffect(() => {
    localStorage.setItem('sam2_labels', JSON.stringify(labels));
  }, [labels]);

  const sanitizedEndpoint = useMemo(() => backendUrl.replace(/\/$/, ''), [backendUrl]);

  // Get color for a label (consistent color per label)
  const getLabelColor = (label: string): string => {
    const index = labels.indexOf(label);
    if (index === -1) return '#000000';
    const hue = (index * 0.618033988749895) % 1.0;
    const r = Math.floor(Math.sin(hue * Math.PI * 2) * 127 + 128);
    const g = Math.floor(Math.sin((hue + 0.33) * Math.PI * 2) * 127 + 128);
    const b = Math.floor(Math.sin((hue + 0.66) * Math.PI * 2) * 127 + 128);
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  };

  // Override segment colors based on labels
  const segmentsWithLabelColors = useMemo(() => {
    return segments.map(seg => ({
      ...seg,
      color: seg.label ? getLabelColor(seg.label) : seg.color,
    }));
  }, [segments, labels]);

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

  // Check if backend is reachable
  const checkBackendHealth = async (): Promise<boolean> => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

    try {
      const response = await fetch(`${sanitizedEndpoint}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response.ok;
    } catch (err) {
      clearTimeout(timeoutId);
      return false;
    }
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!imageFile) {
      setError('Select an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    // Check backend health first
    const isHealthy = await checkBackendHealth();
    if (!isHealthy) {
      setError(`Backend server is not responding at ${sanitizedEndpoint}. Please check the URL and try again.`);
      setIsLoading(false);
      return;
    }

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
      image.onload = () => drawMasks(canvas, image, segmentsWithLabelColors, highlightedSegmentId);
      return;
    }

    drawMasks(canvas, image, segmentsWithLabelColors, highlightedSegmentId);
  }, [segments, segmentsWithLabelColors, imageUrl, highlightedSegmentId]);

  // Sync overlay canvas dimensions with image
  useEffect(() => {
    const overlayCanvas = overlayCanvasRef.current;
    const image = imageRef.current;

    if (!overlayCanvas || !image || !image.complete) {
      return;
    }

    overlayCanvas.width = image.naturalWidth;
    overlayCanvas.height = image.naturalHeight;
  }, [imageUrl]);

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

  // Convert canvas coordinates to image coordinates
  const canvasToImageCoords = (canvasX: number, canvasY: number): {x: number; y: number} => {
    const canvas = overlayCanvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) {
      return {x: canvasX, y: canvasY};
    }

    const rect = canvas.getBoundingClientRect();
    const scaleX = image.naturalWidth / rect.width;
    const scaleY = image.naturalHeight / rect.height;

    return {
      x: (canvasX - rect.left) * scaleX,
      y: (canvasY - rect.top) * scaleY,
    };
  };

  // Get which segment is at a given image coordinate
  const getSegmentAtPixel = (imageX: number, imageY: number): number | string | null => {
    const image = imageRef.current;
    if (!image) {
      return null;
    }

    // Check segments in reverse order (top to bottom in rendering)
    for (let i = segmentsWithLabelColors.length - 1; i >= 0; i--) {
      const segment = segmentsWithLabelColors[i];
      const rle = {
        counts: segment.segmentation.counts,
        size: segment.segmentation.size,
      };

      try {
        const decoded = decode([rle]);
        const maskData = decoded.data as Uint8Array;
        const maskHeight = decoded.shape[0];
        const maskWidth = decoded.shape[1];

        // Convert image coordinates to mask coordinates
        const scaleX = maskWidth / image.naturalWidth;
        const scaleY = maskHeight / image.naturalHeight;
        const maskX = Math.floor(imageX * scaleX);
        const maskY = Math.floor(imageY * scaleY);

        // Check if coordinates are within mask bounds
        if (maskX < 0 || maskX >= maskWidth || maskY < 0 || maskY >= maskHeight) {
          continue;
        }

        // Convert from Fortran-order (column-major) to check pixel
        const maskIndex = maskX * maskHeight + maskY;
        if (maskIndex >= 0 && maskIndex < maskData.length && maskData[maskIndex] !== 0) {
          return segment.id;
        }
      } catch (e) {
        // Skip segments with decode errors
        continue;
      }
    }

    return null;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTab !== 'refine') {
      return;
    }

    // In refine mode: require a selected segment; in create mode: always allow drawing
    if (refinementMode === 'refine' && selectedSegmentId === null) {
      return;
    }

    const coords = canvasToImageCoords(e.clientX, e.clientY);
    setBboxStart(coords);
    setBboxCurrent(coords);
    setIsDrawing(true);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !bboxStart) {
      return;
    }

    const coords = canvasToImageCoords(e.clientX, e.clientY);
    setBboxCurrent(coords);
  };

  const handleMouseUp = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !bboxStart || !imageFile) {
      setIsDrawing(false);
      return;
    }

    const coords = canvasToImageCoords(e.clientX, e.clientY);
    setIsDrawing(false);

    // Calculate bbox [x, y, width, height]
    const x = Math.min(bboxStart.x, coords.x);
    const y = Math.min(bboxStart.y, coords.y);
    const width = Math.abs(coords.x - bboxStart.x);
    const height = Math.abs(coords.y - bboxStart.y);

    // Require minimum bbox size
    if (width < 10 || height < 10) {
      setBboxStart(null);
      setBboxCurrent(null);
      return;
    }

    const bbox = [x, y, width, height];

    // Handle refine vs create mode
    if (refinementMode === 'refine') {
      if (!selectedSegmentId) {
        setError('Please select a segment to refine');
        return;
      }
      handleRefineSegment(bbox);
    } else {
      handleCreateMask(bbox);
    }
  };

  const handleRefineSegment = async (bbox: number[]) => {
    if (!selectedSegmentId || !imageFile) {
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('bbox', JSON.stringify(bbox));
    formData.append('segment_id', String(selectedSegmentId));
    formData.append('segments', JSON.stringify(segments));
    formData.append('model_size', modelSize);

    try {
      const response = await fetch(`${sanitizedEndpoint}/refine-segment`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Refinement request failed');
      }

      const data = await response.json();

      // Update segments with hierarchical IDs
      setSegments(prevSegments => {
        const parentId = selectedSegmentId;
        const filtered = prevSegments.filter(s => s.id !== selectedSegmentId);
        const updated = [...filtered];

        // Find existing child count for this parent
        const childCount = updated.filter(s =>
          String(s.parent_id) === String(parentId)
        ).length;

        if (data.refined_segment) {
          // Generate hierarchical ID: parent.childNumber
          const childId = `${parentId}.${childCount + 1}`;
          updated.push({
            ...data.refined_segment,
            id: childId,
            parent_id: parentId,
          });
        }

        if (data.remainder_segment) {
          // Remainder gets next child ID
          const childId = `${parentId}.${childCount + 2}`;
          updated.push({
            ...data.remainder_segment,
            id: childId,
            parent_id: parentId,
          });
        }

        return updated;
      });

      // Clear selection and bbox
      setSelectedSegmentId(null);
      setBboxStart(null);
      setBboxCurrent(null);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Refinement failed');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateMask = async (bbox: number[]) => {
    if (!imageFile) {
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('bbox', JSON.stringify(bbox));
    formData.append('model_size', modelSize);

    try {
      const response = await fetch(`${sanitizedEndpoint}/create-mask`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Create mask request failed');
      }

      const data = await response.json();

      // Add new mask with fresh sequential ID
      setSegments(prevSegments => {
        // Find the max numeric ID
        const numericIds = prevSegments
          .filter(s => !String(s.id).includes('.'))
          .map(s => parseInt(String(s.id), 10))
          .filter(id => !isNaN(id));
        const maxId = numericIds.length > 0 ? Math.max(...numericIds) : 0;
        const newId = maxId + 1;

        if (data.masks && data.masks.length > 0) {
          // Use the first/best mask from response
          const mask = data.masks[0];
          return [...prevSegments, {
            ...mask,
            id: newId,
          }];
        }

        return prevSegments;
      });

      // Clear bbox
      setBboxStart(null);
      setBboxCurrent(null);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Create mask failed');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Handle labeling via direct image click
    if (!selectedLabel || activeTab !== 'segments' || isDrawing) {
      return;
    }

    const coords = canvasToImageCoords(e.clientX, e.clientY);
    const segmentId = getSegmentAtPixel(coords.x, coords.y);

    if (segmentId !== null) {
      assignLabelToSegment(segmentId, selectedLabel);
    }
  };

  // Draw bbox overlay
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !bboxStart || !bboxCurrent) {
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const image = imageRef.current;
    if (!image) {
      return;
    }

    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bbox
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);

    const x = Math.min(bboxStart.x, bboxCurrent.x);
    const y = Math.min(bboxStart.y, bboxCurrent.y);
    const width = Math.abs(bboxCurrent.x - bboxStart.x);
    const height = Math.abs(bboxCurrent.y - bboxStart.y);

    ctx.strokeRect(x, y, width, height);
  }, [bboxStart, bboxCurrent]);

  const deleteSegment = (segmentId: number | string) => {
    setSegments(prevSegments =>
      prevSegments.filter(s => s.id !== segmentId)
    );
    if (selectedSegmentId === segmentId) {
      setSelectedSegmentId(null);
    }
    if (highlightedSegmentId === segmentId) {
      setHighlightedSegmentId(null);
    }
  };

  const handleSegmentClick = (segmentId: number | string) => {
    if (activeTab === 'refine') {
      setSelectedSegmentId(segmentId);
      // Clear any previous bbox
      setBboxStart(null);
      setBboxCurrent(null);
    } else if (selectedLabel && activeTab === 'segments') {
      // Paint bucket mode: assign label to segment
      assignLabelToSegment(segmentId, selectedLabel);
    }
  };

  const handleSegmentCardClick = (segmentId: number | string) => {
    if (activeTab === 'segments') {
      setHighlightedSegmentId(highlightedSegmentId === segmentId ? null : segmentId);
      // Scroll to show the mask
      if (highlightedSegmentId !== segmentId) {
        scrollToSegment(segmentId);
      }
    }
  };

  const scrollToSegment = (segmentId: number | string) => {
    const segment = segments.find(s => s.id === segmentId);
    if (!segment || !segment.bbox) return;

    const viewer = document.querySelector('.viewer');
    const canvas = canvasRef.current;
    if (!viewer || !canvas) return;

    // segment.bbox = [x, y, width, height]
    const [x, y, width, height] = segment.bbox;
    const centerX = x + width / 2;
    const centerY = y + height / 2;

    // Get the canvas display dimensions (not natural size)
    const rect = canvas.getBoundingClientRect();
    const viewerRect = viewer.getBoundingClientRect();

    // Scale from image coordinates to canvas display coordinates
    const scaleX = rect.width / canvas.naturalWidth;
    const scaleY = rect.height / canvas.naturalHeight;

    const displayCenterX = centerX * scaleX;
    const displayCenterY = centerY * scaleY;

    // Calculate scroll to center the mask in the viewport
    const scrollLeft = displayCenterX - viewerRect.width / 2 + rect.left;
    const scrollTop = displayCenterY - viewerRect.height / 2 + rect.top;

    viewer.scrollLeft = scrollLeft;
    viewer.scrollTop = scrollTop;
  };

  const assignLabelToSegment = (segmentId: number | string, label: string) => {
    setSegments(prevSegments =>
      prevSegments.map(seg =>
        seg.id === segmentId ? {...seg, label} : seg
      )
    );
  };

  const clearSegmentLabel = (segmentId: number | string) => {
    setSegments(prevSegments =>
      prevSegments.map(seg =>
        seg.id === segmentId ? {...seg, label: undefined} : seg
      )
    );
  };

  const addLabel = () => {
    const trimmed = newLabelInput.trim();
    if (trimmed && !labels.includes(trimmed)) {
      setLabels(prev => [...prev, trimmed]);
      setNewLabelInput('');
    }
  };

  const removeLabel = (label: string) => {
    setLabels(prev => prev.filter(l => l !== label));
    if (selectedLabel === label) {
      setSelectedLabel(null);
    }
    // Remove this label from all segments
    setSegments(prevSegments =>
      prevSegments.map(seg =>
        seg.label === label ? {...seg, label: undefined} : seg
      )
    );
  };

  // Save project function
  const saveProject = useCallback(async () => {
    if (!responseMetadata || segments.length === 0) {
      return;
    }

    setIsSaving(true);
    setSaveError(null);

    const formData = new FormData();
    formData.append('data', JSON.stringify({
      image: responseMetadata.image,
      image_filename: imageFile?.name || 'image.jpg',
      segments: segments,
      labels: labels,
    }));

    try {
      const response = await fetch(`${sanitizedEndpoint}/save-project`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Save request failed');
      }

      const data = await response.json();
      setLastSaved(new Date());
      console.log('Project saved:', data);
    } catch (err) {
      if (err instanceof Error) {
        setSaveError(err.message);
      } else {
        setSaveError('Save failed');
      }
    } finally {
      setIsSaving(false);
    }
  }, [responseMetadata, segments, labels, imageFile, sanitizedEndpoint]);

  // Auto-save logic with debounce
  useEffect(() => {
    // Clear existing timeout
    if (autoSaveTimeoutRef.current) {
      window.clearTimeout(autoSaveTimeoutRef.current);
    }

    // Only auto-save if we have segments
    if (segments.length === 0) {
      return;
    }

    // Set up new timeout for auto-save (30 seconds)
    autoSaveTimeoutRef.current = window.setTimeout(() => {
      saveProject();
    }, 30000);

    return () => {
      if (autoSaveTimeoutRef.current) {
        window.clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, [segments, labels, saveProject]);

  // Download COCO JSON
  const downloadCocoJson = () => {
    if (!responseMetadata || segments.length === 0) {
      return;
    }

    // Build COCO format
    const categories = labels.map((label, idx) => ({
      id: idx + 1,
      name: label,
      supercategory: 'object',
    }));

    const labelToId: Record<string, number> = {};
    labels.forEach((label, idx) => {
      labelToId[label] = idx + 1;
    });

    const annotations = segments
      .filter(seg => seg.label && labelToId[seg.label])
      .map((seg, idx) => ({
        id: idx + 1,
        image_id: 1,
        category_id: labelToId[seg.label!],
        bbox: seg.bbox,
        area: seg.area,
        segmentation: seg.segmentation,
        iscrowd: 0,
        score: seg.predicted_iou,
      }));

    const cocoFormat = {
      images: [{
        id: 1,
        file_name: imageFile?.name || 'image.jpg',
        width: responseMetadata.image.width,
        height: responseMetadata.image.height,
      }],
      annotations,
      categories,
    };

    const blob = new Blob([JSON.stringify(cocoFormat, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'coco_annotations.json';
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

            <hr />
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
              <canvas
                ref={overlayCanvasRef}
                className="overlay-canvas"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onClick={handleImageClick}
                style={{
                  cursor: activeTab === 'refine' && selectedSegmentId !== null ? 'crosshair' : selectedLabel && activeTab === 'segments' ? 'pointer' : 'default',
                }}
              />
            </div>
          ) : (
            <div className="placeholder">Upload an image to see the preview.</div>
          )}
        </section>

        <section className="panel segments-panel">
          {/* Tab Navigation */}
          <div className="tabs">
            <button
              type="button"
              className={`tab-button ${activeTab === 'segments' ? 'active' : ''}`}
              onClick={() => {
                setActiveTab('segments');
                setSelectedSegmentId(null);
                setBboxStart(null);
                setBboxCurrent(null);
              }}
              disabled={segments.length === 0}
            >
              Segments ({segments.length})
            </button>
            <button
              type="button"
              className={`tab-button ${activeTab === 'refine' ? 'active' : ''}`}
              onClick={() => setActiveTab('refine')}
              disabled={segments.length === 0}
            >
              Refine
            </button>
          </div>

          {/* Segments Tab Content */}
          <div className={`tab-content ${activeTab === 'segments' ? 'active' : ''}`}>
            <div className="panel-header">
              <h2>Segments & Labels</h2>
              <div style={{display: 'flex', gap: '0.5rem', flexWrap: 'wrap'}}>
                <button
                  onClick={saveProject}
                  disabled={isSaving || segments.length === 0}
                  style={{
                    backgroundColor: '#10b981',
                    fontSize: '1.05rem',
                    fontWeight: 600,
                    padding: '0.75rem 1.5rem',
                  }}
                >
                  {isSaving ? 'Saving...' : '💾 Save Project'}
                </button>
                <button onClick={downloadCocoJson} disabled={segments.length === 0}>
                  Download COCO JSON
                </button>
                <button onClick={downloadJson} disabled={!responseMetadata}>
                  Download RAW JSON
                </button>
              </div>
            </div>

            {lastSaved && (
              <p className="hint" style={{color: '#10b981'}}>
                ✓ Last saved: {lastSaved.toLocaleTimeString()} (Auto-save in 30s)
              </p>
            )}
            {saveError && <p className="error">Save failed: {saveError}</p>}

            {responseMetadata?.model_size && (
              <p className="hint">Generated with model: {responseMetadata.model_size}</p>
            )}

            {/* Labels Section */}
            <div style={{marginBottom: '1.5rem'}}>
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem'}}>
                <strong>Labels</strong>
                <button
                  type="button"
                  onClick={() => setShowLabelSettings(!showLabelSettings)}
                  style={{fontSize: '0.85rem', padding: '0.35rem 0.75rem'}}
                >
                  {showLabelSettings ? 'Hide' : 'Manage'}
                </button>
              </div>

              {showLabelSettings && (
                <div style={{marginBottom: '1rem', padding: '0.75rem', background: '#f8fafc', borderRadius: '8px'}}>
                  <div style={{display: 'flex', gap: '0.5rem', marginBottom: '0.5rem'}}>
                    <input
                      type="text"
                      value={newLabelInput}
                      onChange={e => setNewLabelInput(e.target.value)}
                      onKeyPress={e => e.key === 'Enter' && addLabel()}
                      placeholder="New label name"
                      style={{flex: 1, padding: '0.4rem', fontSize: '0.9rem', borderRadius: '6px', border: '1px solid #cbd5e1'}}
                    />
                    <button
                      type="button"
                      onClick={addLabel}
                      style={{padding: '0.4rem 0.75rem', fontSize: '0.9rem'}}
                    >
                      Add
                    </button>
                  </div>
                  <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.5rem'}}>
                    {labels.map(label => (
                      <div
                        key={label}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem',
                          padding: '0.35rem 0.6rem',
                          background: '#fff',
                          border: '1px solid #e2e8f0',
                          borderRadius: '6px',
                          fontSize: '0.9rem',
                        }}
                      >
                        <span>{label}</span>
                        <button
                          type="button"
                          onClick={() => removeLabel(label)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#dc2626',
                            cursor: 'pointer',
                            padding: '0',
                            fontSize: '0.9rem',
                            lineHeight: 1,
                          }}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.75rem'}}>
                {labels.map(label => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => setSelectedLabel(selectedLabel === label ? null : label)}
                    className={selectedLabel === label ? 'label-button selected' : 'label-button'}
                    style={{
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.9rem',
                      background: selectedLabel === label ? getLabelColor(label) : '#fff',
                      color: selectedLabel === label ? '#fff' : '#334155',
                      border: `2px solid ${getLabelColor(label)}`,
                      borderRadius: '8px',
                      cursor: 'pointer',
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {selectedLabel && (
                <p className="hint">
                  Label selected: <strong>{selectedLabel}</strong>. Click on the image over a segment or click segment cards to apply this label.
                </p>
              )}
            </div>

            {/* Segments Grid */}
            {segments.length === 0 ? (
              <p className="hint">Run segmentation to populate mask proposals.</p>
            ) : (
              <div className="segment-grid">
                {segments.map(segment => {
                  const displayColor = segment.label ? getLabelColor(segment.label) : segment.color;
                  const isHighlighted = highlightedSegmentId === segment.id;
                  return (
                    <article
                      key={segment.id}
                      className={`segment-card ${selectedLabel ? 'clickable' : ''}`}
                      onClick={() => {
                        handleSegmentCardClick(segment.id);
                        handleSegmentClick(segment.id);
                      }}
                      style={{
                        cursor: selectedLabel ? 'pointer' : 'default',
                        backgroundColor: isHighlighted ? '#f0f4ff' : 'transparent',
                        border: isHighlighted ? `2px solid ${displayColor}` : '1px solid #e2e8f0',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <div className="swatch" style={{backgroundColor: displayColor}} />
                      <div style={{flex: 1}}>
                        <p className="segment-title">Mask #{segment.id}</p>
                        {segment.label && (
                          <p style={{margin: '0.25rem 0', fontSize: '0.85rem', fontWeight: 600, color: displayColor}}>
                            Label: {segment.label}
                          </p>
                        )}
                        <p className="segment-meta">
                          area: {segment.area.toFixed(0)} px² • IoU:{' '}
                          {segment.predicted_iou.toFixed(3)}
                        </p>
                      </div>
                      <div style={{display: 'flex', gap: '0.25rem', alignItems: 'center'}}>
                        {segment.label && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              clearSegmentLabel(segment.id);
                            }}
                            style={{
                              background: 'none',
                              border: 'none',
                              color: '#dc2626',
                              cursor: 'pointer',
                              fontSize: '1.2rem',
                              padding: '0.25rem',
                              lineHeight: 1,
                            }}
                            title="Clear label"
                          >
                            ×
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSegment(segment.id);
                          }}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#7c3aed',
                            cursor: 'pointer',
                            fontSize: '1.2rem',
                            padding: '0.25rem',
                            lineHeight: 1,
                          }}
                          title="Delete mask"
                        >
                          🗑️
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>

          {/* Refine Tab Content */}
          <div className={`tab-content ${activeTab === 'refine' ? 'active' : ''}`}>
            <div style={{marginBottom: '1.5rem'}}>
              <h3 style={{margin: '0 0 1rem 0'}}>Refinement Mode</h3>

              {segments.length === 0 ? (
                <p className="hint">Run segmentation first to refine masks.</p>
              ) : (
                <>
                  {/* Mode Toggle Buttons */}
                  <div style={{marginBottom: '1rem', display: 'flex', gap: '0.5rem'}}>
                    <button
                      type="button"
                      onClick={() => {
                        setRefinementMode('refine');
                        setSelectedSegmentId(null);
                      }}
                      style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: refinementMode === 'refine' ? '#3b82f6' : '#e2e8f0',
                        color: refinementMode === 'refine' ? '#fff' : '#334155',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: refinementMode === 'refine' ? 600 : 400,
                      }}
                    >
                      Refine Mask
                    </button>
                    <button
                      type="button"
                      onClick={() => setRefinementMode('create')}
                      style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: refinementMode === 'create' ? '#10b981' : '#e2e8f0',
                        color: refinementMode === 'create' ? '#fff' : '#334155',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: refinementMode === 'create' ? 600 : 400,
                      }}
                    >
                      Create New Mask
                    </button>
                  </div>

                  <p className="hint">
                    {refinementMode === 'refine'
                      ? selectedSegmentId
                        ? `✓ Segment ${selectedSegmentId} selected. Draw a box on the image to refine it.`
                        : 'Click a segment card below to select it, then draw a bounding box on the image to refine it.'
                      : 'Draw a bounding box on the image to create a new independent mask in that region.'}
                  </p>

                  <div style={{marginTop: '1.5rem'}}>
                    <h4 style={{margin: '0 0 0.75rem 0', fontSize: '0.95rem'}}>Segments</h4>
                    <div className="segment-grid">
                      {segments.map(segment => {
                        const displayColor = segment.label ? getLabelColor(segment.label) : segment.color;
                        return (
                          <article
                            key={segment.id}
                            className={`segment-card ${selectedSegmentId === segment.id ? 'selected' : ''} clickable`}
                            onClick={() => handleSegmentClick(segment.id)}
                            style={{
                              cursor: 'pointer',
                            }}
                          >
                            <div className="swatch" style={{backgroundColor: displayColor}} />
                            <div style={{flex: 1}}>
                              <p className="segment-title">Mask #{segment.id}</p>
                              {segment.label && (
                                <p style={{margin: '0.25rem 0', fontSize: '0.85rem', fontWeight: 600, color: displayColor}}>
                                  Label: {segment.label}
                                </p>
                              )}
                              <p className="segment-meta">
                                area: {segment.area.toFixed(0)} px² • IoU:{' '}
                                {segment.predicted_iou.toFixed(3)}
                              </p>
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function drawMasks(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  segments: Segment[],
  highlightedSegmentId?: number | string | null
) {
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

  const overlay = ctx.createImageData(width, height);

  segments.forEach(segment => {
    const rle: RLEObject = {
      counts: segment.segmentation.counts,
      size: segment.segmentation.size,
    };

    const decoded = decode([rle]);
    const maskData = decoded.data as Uint8Array;
    const maskHeight = decoded.shape[0];
    const maskWidth = decoded.shape[1];
    const maskRowMajor = new Uint8Array(maskHeight * maskWidth);

    for (let y = 0; y < maskHeight; y++) {
      for (let x = 0; x < maskWidth; x++) {
        const srcIndex = x * maskHeight + y; // convert Fortran-order to row-major
        const dstIndex = y * maskWidth + x;
        maskRowMajor[dstIndex] = maskData[srcIndex];
      }
    }

    const [r, g, b] = hexToRgb(segment.color);
    const scaleX = maskWidth / width;
    const scaleY = maskHeight / height;

    // Use higher opacity for highlighted mask, normal for others
    const isHighlighted = highlightedSegmentId !== null && segment.id === highlightedSegmentId;
    const alpha = isHighlighted ? 210 : 140; // Brighter overlay for highlighted

    for (let y = 0; y < height; y++) {
      const srcY = Math.min(Math.floor(y * scaleY), maskHeight - 1);
      for (let x = 0; x < width; x++) {
        const srcX = Math.min(Math.floor(x * scaleX), maskWidth - 1);
        const srcIndex = srcY * maskWidth + srcX;
        if (maskRowMajor[srcIndex] === 0) {
          continue;
        }
        const dstIndex = (y * width + x) * 4;
        overlay.data[dstIndex] = r;
        overlay.data[dstIndex + 1] = g;
        overlay.data[dstIndex + 2] = b;
        overlay.data[dstIndex + 3] = alpha;
      }
    }
  });

  ctx.putImageData(overlay, 0, 0);
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
