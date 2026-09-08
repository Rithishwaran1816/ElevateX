/**
 * ElevateX Three.js 3D Web Viewer & Photogrammetry Dashboard Controller.
 */

// State
let uploadedVideoPath = null;
let uploadedTelemetryPath = null;
let pollingInterval = null;
let isMeasuring = false;
let measurePoints = [];
let measureMarkers = [];
let measureLine = null;

// Three.js Globals
let scene, camera, renderer, controls;
let currentModelObject = null;
let raycaster, mouse;
let pointMaterial = null;

document.addEventListener("DOMContentLoaded", () => {
    initThreeViewer();
    initUploadHandlers();
    initToolbarControls();
    initTabHandlers();
    checkSystemStatus();
});

/* -----------------------------------------------------------------
 * 1. Three.js Viewport Initialization
 * ----------------------------------------------------------------- */
function initThreeViewer() {
    const container = document.getElementById("canvas-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060911);

    // Camera (Y-up convention)
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.up.set(0, 1, 0);
    camera.position.set(0, 5, 10);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(10, 20, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x90b0ff, 0.4);
    dirLight2.position.set(-10, -10, -10);
    scene.add(dirLight2);

    // Helpers
    const grid = new THREE.GridHelper(30, 30, 0x3b82f6, 0x1f2937);
    grid.position.y = -0.01;
    scene.add(grid);

    // Raycaster for Measurement
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.1;
    mouse = new THREE.Vector2();

    // Window Resize
    window.addEventListener("resize", onWindowResize);

    // Canvas Click for Measurement
    renderer.domElement.addEventListener("click", onCanvasClick);

    // Animation Loop
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById("canvas-container");
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

/* -----------------------------------------------------------------
 * 2. Model Loading (GLB / PLY / OBJ)
 * ----------------------------------------------------------------- */
function load3DModel(type = "textured") {
    const placeholder = document.getElementById("viewer-placeholder");
    if (placeholder) placeholder.classList.add("hidden");

    // Remove existing model
    if (currentModelObject) {
        scene.remove(currentModelObject);
        currentModelObject = null;
    }
    clearMeasurement();

    const timestamp = new Date().getTime();

    if (type === "textured" || type === "wireframe") {
        // Try GLB first
        const gltfLoader = new THREE.GLTFLoader();
        gltfLoader.load(
            `/api/models/model.glb?t=${timestamp}`,
            (gltf) => {
                currentModelObject = gltf.scene;
                if (type === "wireframe") {
                    currentModelObject.traverse((child) => {
                        if (child.isMesh) child.material.wireframe = true;
                    });
                }
                fitObjectToView(currentModelObject);
                scene.add(currentModelObject);
            },
            undefined,
            (err) => {
                // Fallback to PLY mesh
                loadPlyMesh(`/api/models/textured_model.ply?t=${timestamp}`, type === "wireframe");
            }
        );
    } else if (type === "dense_pcd") {
        loadPlyPointCloud(`/api/models/dense_points_clean.ply?t=${timestamp}`);
    } else if (type === "sparse_pcd") {
        loadPlyPointCloud(`/api/models/sparse_points.ply?t=${timestamp}`);
    }
}

function loadPlyMesh(url, wireframe = false) {
    const plyLoader = new THREE.PLYLoader();
    plyLoader.load(url, (geometry) => {
        geometry.computeVertexNormals();
        let material;
        if (geometry.hasAttribute("color")) {
            material = new THREE.MeshStandardMaterial({
                vertexColors: true,
                roughness: 0.6,
                metalness: 0.1,
                wireframe: wireframe,
                side: THREE.DoubleSide
            });
        } else {
            material = new THREE.MeshStandardMaterial({
                color: 0x94a3b8,
                roughness: 0.6,
                wireframe: wireframe,
                side: THREE.DoubleSide
            });
        }
        const mesh = new THREE.Mesh(geometry, material);
        currentModelObject = mesh;
        fitObjectToView(currentModelObject);
        scene.add(currentModelObject);
    });
}

function loadPlyPointCloud(url) {
    const plyLoader = new THREE.PLYLoader();
    const pointSize = parseFloat(document.getElementById("slider-point-size").value) * 0.04;

    plyLoader.load(url, (geometry) => {
        const hasColors = geometry.hasAttribute("color");
        pointMaterial = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: hasColors,
            color: hasColors ? 0xffffff : 0x38bdf8
        });
        const points = new THREE.Points(geometry, pointMaterial);
        currentModelObject = points;
        fitObjectToView(currentModelObject);
        scene.add(currentModelObject);
    });
}

function fitObjectToView(object) {
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Center object
    object.position.x = -center.x;
    object.position.y = -center.y + (size.y / 2);
    object.position.z = -center.z;

    // Adjust Camera
    camera.position.set(maxDim * 1.2, maxDim * 1.0, maxDim * 1.5);
    camera.lookAt(0, size.y / 4, 0);
    controls.target.set(0, size.y / 4, 0);
    controls.update();
}

/* -----------------------------------------------------------------
 * 3. Point-to-Point Measurement Tool
 * ----------------------------------------------------------------- */
function onCanvasClick(event) {
    if (!isMeasuring || !currentModelObject) return;

    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(currentModelObject, true);

    if (intersects.length > 0) {
        const point = intersects[0].point;
        addMeasurePoint(point);
    }
}

function addMeasurePoint(point) {
    measurePoints.push(point);

    // Add sphere marker
    const markerGeom = new THREE.SphereGeometry(0.08, 16, 16);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const marker = new THREE.Mesh(markerGeom, markerMat);
    marker.position.copy(point);
    scene.add(marker);
    measureMarkers.push(marker);

    const hud = document.getElementById("measurement-hud");
    hud.classList.remove("hidden");

    if (measurePoints.length === 1) {
        document.getElementById("m-p1").textContent = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`;
        document.getElementById("m-p2").textContent = "Select Point 2...";
        document.getElementById("m-dist").textContent = "-";
    } else if (measurePoints.length === 2) {
        const p1 = measurePoints[0];
        const p2 = measurePoints[1];
        document.getElementById("m-p2").textContent = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`;

        const dist = p1.distanceTo(p2);
        document.getElementById("m-dist").textContent = `${dist.toFixed(3)} units (meters)`;

        // Draw line between points
        const lineGeom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 });
        measureLine = new THREE.Line(lineGeom, lineMat);
        scene.add(measureLine);
    } else {
        // Reset and start new measurement
        clearMeasurement();
        addMeasurePoint(point);
    }
}

function clearMeasurement() {
    measurePoints = [];
    measureMarkers.forEach((m) => scene.remove(m));
    measureMarkers = [];
    if (measureLine) {
        scene.remove(measureLine);
        measureLine = null;
    }
    const hud = document.getElementById("measurement-hud");
    if (hud && !isMeasuring) hud.classList.add("hidden");
}

/* -----------------------------------------------------------------
 * 4. UI Handlers & Pipeline Polling
 * ----------------------------------------------------------------- */
function initUploadHandlers() {
    const uploadForm = document.getElementById("upload-form");
    const videoInput = document.getElementById("video-input");
    const telInput = document.getElementById("telemetry-input");
    const videoDropZone = document.getElementById("video-drop-zone");
    const telDropZone = document.getElementById("tel-drop-zone");
    const videoFilename = document.getElementById("video-filename");
    const telFilename = document.getElementById("tel-filename");
    const btnStart = document.getElementById("btn-start-process");
    const btnDemoLoad = document.getElementById("btn-demo-load");
    const progressContainer = document.getElementById("upload-progress-container");
    const progressFill = document.getElementById("upload-progress-fill");
    const progressText = document.getElementById("upload-progress-text");

    // Helper to format file size
    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    // File input changes
    videoInput.addEventListener("change", () => {
        if (videoInput.files.length > 0) {
            const file = videoInput.files[0];
            videoFilename.innerHTML = `<strong>Selected:</strong> ${file.name} (${formatBytes(file.size)})`;
        }
    });

    telInput.addEventListener("change", () => {
        if (telInput.files.length > 0) {
            const file = telInput.files[0];
            telFilename.innerHTML = `<strong>Selected:</strong> ${file.name} (${formatBytes(file.size)})`;
        }
    });

    // Drag & Drop for Video
    ['dragenter', 'dragover'].forEach(eventName => {
        videoDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoDropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        videoDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoDropZone.classList.remove('dragover');
        }, false);
    });

    videoDropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            videoInput.files = files;
            const file = files[0];
            videoFilename.innerHTML = `<strong>Selected:</strong> ${file.name} (${formatBytes(file.size)})`;
        }
    });

    // Drag & Drop for Telemetry
    ['dragenter', 'dragover'].forEach(eventName => {
        telDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            telDropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        telDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            telDropZone.classList.remove('dragover');
        }, false);
    });

    telDropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            telInput.files = files;
            const file = files[0];
            telFilename.innerHTML = `<strong>Selected:</strong> ${file.name} (${formatBytes(file.size)})`;
        }
    });

    // Demo Flight Quick-Load Button
    if (btnDemoLoad) {
        btnDemoLoad.addEventListener("click", async () => {
            btnDemoLoad.disabled = true;
            btnDemoLoad.innerHTML = `<span class="badge badge-running">Loading...</span>`;
            try {
                const res = await fetch("/api/sample/load", { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    uploadedVideoPath = data.video_path;
                    uploadedTelemetryPath = data.telemetry_path;
                    videoFilename.innerHTML = `<strong>Demo Video:</strong> drone_flight_sample.mp4`;
                    if (telFilename) telFilename.innerHTML = `<strong>Demo Telemetry:</strong> flight_telemetry_sample.csv`;

                    document.getElementById("video-details").classList.remove("hidden");
                    document.getElementById("meta-res").textContent = data.video_metadata.resolution;
                    document.getElementById("meta-fps").textContent = data.video_metadata.fps;
                    document.getElementById("meta-dur").textContent = data.video_metadata.duration_formatted;
                    document.getElementById("meta-frames").textContent = data.video_metadata.total_frames;

                    updateSensorStatus(data.sensor_status);
                    btnStart.disabled = false;
                    btnDemoLoad.innerHTML = `Demo Loaded`;
                } else {
                    alert(`Failed to load demo: ${data.error}`);
                    btnDemoLoad.innerHTML = `Demo Flight`;
                    btnDemoLoad.disabled = false;
                }
            } catch (err) {
                alert(`Error loading demo: ${err.message}`);
                btnDemoLoad.innerHTML = `Demo Flight`;
                btnDemoLoad.disabled = false;
            }
        });
    }

    // Form Upload Submission via XMLHttpRequest for live upload progress
    uploadForm.addEventListener("submit", (e) => {
        e.preventDefault();
        if (!videoInput.files || !videoInput.files.length) {
            alert("Please select or drop a drone video file first.");
            return;
        }

        const formData = new FormData();
        formData.append("video", videoInput.files[0]);
        if (telInput.files && telInput.files.length) {
            formData.append("telemetry", telInput.files[0]);
        }

        const btnUpload = document.getElementById("btn-upload");
        btnUpload.disabled = true;
        btnUpload.innerHTML = `Uploading...`;

        if (progressContainer) {
            progressContainer.classList.remove("hidden");
            progressFill.style.width = "0%";
            progressText.textContent = "Uploading: 0%";
        }

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/upload", true);

        // Upload progress
        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable && progressContainer) {
                const percent = Math.round((event.loaded / event.total) * 100);
                progressFill.style.width = `${percent}%`;
                progressText.textContent = `Uploading: ${percent}% (${formatBytes(event.loaded)} / ${formatBytes(event.total)})`;
            }
        };

        xhr.onload = () => {
            btnUpload.disabled = false;
            btnUpload.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg> Upload & Inspect`;

            try {
                const data = JSON.parse(xhr.responseText);
                if (xhr.status === 200 && data.success) {
                    uploadedVideoPath = data.video_path;
                    uploadedTelemetryPath = data.telemetry_path;

                    if (progressContainer) {
                        progressFill.style.width = "100%";
                        progressText.textContent = "Upload & Inspection Complete (100%)";
                    }

                    // Show video details
                    document.getElementById("video-details").classList.remove("hidden");
                    document.getElementById("meta-res").textContent = data.video_metadata.resolution;
                    document.getElementById("meta-fps").textContent = data.video_metadata.fps;
                    document.getElementById("meta-dur").textContent = data.video_metadata.duration_formatted;
                    document.getElementById("meta-frames").textContent = data.video_metadata.total_frames;

                    // Update Sensor pills
                    updateSensorStatus(data.sensor_status);

                    btnStart.disabled = false;
                } else {
                    alert(`Upload error (${xhr.status}): ${data.error || 'Server error'}`);
                    if (progressContainer) progressContainer.classList.add("hidden");
                }
            } catch (err) {
                alert(`Failed to parse response: ${xhr.responseText}`);
                if (progressContainer) progressContainer.classList.add("hidden");
            }
        };

        xhr.onerror = () => {
            btnUpload.disabled = false;
            btnUpload.innerHTML = `Upload & Inspect`;
            alert("Network connection error while uploading file.");
            if (progressContainer) progressContainer.classList.add("hidden");
        };

        xhr.send(formData);
    });

    btnStart.addEventListener("click", async () => {
        if (!uploadedVideoPath) return;

        btnStart.disabled = true;
        btnStart.innerHTML = `<span class="badge badge-running">RUNNING</span> Processing 3D Pipeline...`;

        const res = await fetch("/api/process/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_path: uploadedVideoPath,
                telemetry_path: uploadedTelemetryPath
            })
        });
        const data = await res.json();
        if (data.success) {
            startStatusPolling();
        } else {
            alert(data.error || "Failed to start pipeline");
            btnStart.disabled = false;
            btnStart.textContent = "Execute 3D Reconstruction";
        }
    });
}

function startStatusPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(pollPipelineStatus, 1000);
}

async function pollPipelineStatus() {
    try {
        const res = await fetch("/api/process/status");
        const statusData = await res.json();

        // Update overall status badge
        const badge = document.getElementById("overall-status-badge");
        badge.textContent = statusData.state;
        badge.className = `badge badge-${statusData.state.toLowerCase()}`;

        // Update each stage item
        const stageMap = {
            "stage1_video": "stage-1",
            "stage2_quality": "stage-2",
            "stage3_sfm": "stage-3",
            "stage4_mesh": "stage-4",
            "stage5_texture": "stage-5",
            "stage6_georef": "stage-6",
            "stage7_assessment": "stage-7"
        };

        for (const [key, stageId] of Object.entries(stageMap)) {
            const st = statusData.stages[key];
            const el = document.getElementById(stageId);
            if (st && el) {
                const sBadge = el.querySelector(".stage-badge");
                sBadge.textContent = st.status;
                sBadge.className = `stage-badge badge-${st.status.toLowerCase()}`;
            }
        }

        // Update sensor statuses
        if (statusData.sensor_availability) {
            updateSensorStatus(statusData.sensor_availability);
        }

        // Update duration
        document.getElementById("stat-time").textContent = `${statusData.elapsed_seconds || 0}s`;

        // Check completion
        if (statusData.state === "COMPLETED") {
            clearInterval(pollingInterval);
            const btnStart = document.getElementById("btn-start-process");
            btnStart.disabled = false;
            btnStart.textContent = "Re-Execute Reconstruction";

            const qReport = statusData.artifacts?.quality_report;
            if (qReport) {
                document.getElementById("stat-vertices").textContent = (qReport.reconstruction_metrics.mesh_vertices || 0).toLocaleString();
                document.getElementById("stat-faces").textContent = (qReport.reconstruction_metrics.mesh_faces || 0).toLocaleString();
                document.getElementById("stat-dense-pts").textContent = (qReport.reconstruction_metrics.dense_points || 0).toLocaleString();
                document.getElementById("stat-georef").textContent = qReport.georeferencing.status;

                document.getElementById("quality-json-viewer").textContent = JSON.stringify(qReport, null, 2);
            }

            // Fetch georef report
            fetchGeorefReport();

            // Load generated 3D Model into Three.js
            load3DModel(document.getElementById("select-render-mode").value);
        } else if (statusData.state === "FAILED") {
            clearInterval(pollingInterval);
            const btnStart = document.getElementById("btn-start-process");
            btnStart.disabled = false;
            btnStart.textContent = "Retry Reconstruction";
            alert(`Reconstruction Pipeline Failed: ${statusData.error}`);
        }
    } catch (e) {
        console.error("Polling error", e);
    }
}

async function fetchGeorefReport() {
    try {
        const res = await fetch("/api/reports/georeference_report.json");
        if (res.ok) {
            const data = await res.json();
            document.getElementById("georef-json-viewer").textContent = JSON.stringify(data, null, 2);
        }
    } catch (e) {}
}

function updateSensorStatus(sensors) {
    if (!sensors) return;
    const mapping = {
        "GPS": "sensor-gps",
        "Flight Metadata": "sensor-telemetry",
        "IMU": "sensor-imu",
        "RTK": "sensor-rtk",
        "Camera Intrinsics": "sensor-intrinsics"
    };

    for (const [key, id] of Object.entries(mapping)) {
        const pill = document.getElementById(id);
        const val = sensors[key];
        if (pill && val) {
            const valEl = pill.querySelector(".val");
            valEl.textContent = val;
            pill.className = `sensor-pill ${val === 'AVAILABLE' ? 'available' : 'unavailable'}`;
        }
    }
}

async function checkSystemStatus() {
    try {
        const res = await fetch("/api/system/status");
        const data = await res.json();
        const badge = document.getElementById("engine-status");
        if (data.colmap_available) {
            badge.textContent = "Engine: COLMAP Photogrammetry";
        } else {
            badge.textContent = "Engine: Native Open3D/OpenCV SfM";
        }
    } catch (e) {}
}

function initToolbarControls() {
    // Mode switcher
    document.getElementById("select-render-mode").addEventListener("change", (e) => {
        load3DModel(e.target.value);
    });

    // Point size slider
    document.getElementById("slider-point-size").addEventListener("input", (e) => {
        if (pointMaterial) {
            pointMaterial.size = parseFloat(e.target.value) * 0.04;
        }
    });

    // Reset Camera
    document.getElementById("btn-reset-cam").addEventListener("click", () => {
        if (currentModelObject) fitObjectToView(currentModelObject);
    });

    // Measurement tool toggle
    const btnMeasure = document.getElementById("btn-measure");
    btnMeasure.addEventListener("click", () => {
        isMeasuring = !isMeasuring;
        btnMeasure.classList.toggle("active", isMeasuring);
        const hud = document.getElementById("measurement-hud");
        if (isMeasuring) {
            hud.classList.remove("hidden");
        } else {
            clearMeasurement();
        }
    });

    // Fullscreen
    document.getElementById("btn-fullscreen").addEventListener("click", () => {
        const elem = document.getElementById("canvas-container");
        if (!document.fullscreenElement) {
            elem.requestFullscreen().catch(err => alert(err.message));
        } else {
            document.exitFullscreen();
        }
    });
}

function initTabHandlers() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabBtns.forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            const content = document.getElementById(target);
            if (content) content.classList.add("active");
        });
    });
}
