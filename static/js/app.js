/**
 * Bicep Curl AI Coach - Real-Time Frontend Logic
 * Captures browser webcam, transmits frames to Flask backend,
 * and renders pose skeletons, kinematic angles, and live form feedback.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('outputCanvas');
  const ctx = canvas.getContext('2d');
  const cameraPlaceholder = document.getElementById('cameraPlaceholder');
  const toggleCameraBtn = document.getElementById('toggleCameraBtn');
  const startCameraBtnInline = document.getElementById('startCameraBtnInline');
  const camBtnText = document.getElementById('camBtnText');
  const camIcon = document.getElementById('camIcon');
  const resetBtn = document.getElementById('resetBtn');
  const soundToggle = document.getElementById('soundToggle');
  const skeletonToggle = document.getElementById('skeletonToggle');

  // HUD Elements
  const liveIndicator = document.getElementById('liveIndicator');
  const viewportStatus = document.getElementById('viewportStatus');
  const fpsDisplay = document.getElementById('fpsDisplay');
  const latencyDisplay = document.getElementById('latencyDisplay');
  const exerciseStateBadge = document.getElementById('exerciseStateBadge');
  const formStatusBadge = document.getElementById('formStatusBadge');
  const hudElbowAngle = document.getElementById('hudElbowAngle');
  const hudRepCount = document.getElementById('hudRepCount');

  // Stats Elements
  const mainRepCount = document.getElementById('mainRepCount');
  const partialRepCount = document.getElementById('partialRepCount');
  const currentPhaseText = document.getElementById('currentPhaseText');
  const romStatusText = document.getElementById('romStatusText');
  const repRingProgress = document.getElementById('repRingProgress');

  // ML Elements
  const confidenceTag = document.getElementById('confidenceTag');
  const mlPredictionTitle = document.getElementById('mlPredictionTitle');
  const mlPredictionDesc = document.getElementById('mlPredictionDesc');
  const predictionBox = document.getElementById('predictionBox');
  const probCorrect = document.getElementById('probCorrect');
  const probCorrectText = document.getElementById('probCorrectText');
  const probPartial = document.getElementById('probPartial');
  const probPartialText = document.getElementById('probPartialText');
  const probSwing = document.getElementById('probSwing');
  const probSwingText = document.getElementById('probSwingText');

  // Metric Elements
  const leftElbowText = document.getElementById('leftElbowText');
  const rightElbowText = document.getElementById('rightElbowText');
  const torsoTiltText = document.getElementById('torsoTiltText');
  const shoulderSymText = document.getElementById('shoulderSymText');
  const cuesList = document.getElementById('cuesList');

  // State Variables
  let isCameraRunning = false;
  let isProcessingFrame = false;
  let animationFrameId = null;
  let stream = null;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 0;

  // Offscreen canvas for fast frame serialization
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
  captureCanvas.width = 480;
  captureCanvas.height = 360;

  // Web Audio Context for Beep on Rep Completion
  let audioCtx = null;
  function playRepChime(success = true) {
    if (!soundToggle.checked) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (success) {
        osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
        osc.frequency.exponentialRampToValueAtTime(880.00, audioCtx.currentTime + 0.15); // A5
        gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.25);
      } else {
        osc.frequency.setValueAtTime(329.63, audioCtx.currentTime); // E4
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.2);
      }
    } catch (e) {
      console.warn('Audio playback error:', e);
    }
  }

  // Camera Management
  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      video.srcObject = stream;
      await video.play();

      // Match canvas dimensions to video
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
      };

      isCameraRunning = true;
      cameraPlaceholder.style.opacity = '0';
      setTimeout(() => { cameraPlaceholder.style.display = 'none'; }, 300);

      camBtnText.textContent = 'Stop Camera';
      camIcon.innerHTML = '<rect x="6" y="6" width="12" height="12"></rect>';
      toggleCameraBtn.classList.remove('btn-primary');
      toggleCameraBtn.classList.add('btn-secondary');

      liveIndicator.classList.add('active');
      viewportStatus.textContent = 'LIVE ANALYSIS';

      processVideoLoop();
    } catch (err) {
      console.error('Webcam access error:', err);
      alert('Unable to access webcam. Please verify camera permissions in your browser.');
    }
  }

  function stopCamera() {
    isCameraRunning = false;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    video.srcObject = null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    cameraPlaceholder.style.display = 'flex';
    setTimeout(() => { cameraPlaceholder.style.opacity = '1'; }, 20);

    camBtnText.textContent = 'Start Camera';
    camIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    toggleCameraBtn.classList.add('btn-primary');
    toggleCameraBtn.classList.remove('btn-secondary');

    liveIndicator.classList.remove('active');
    viewportStatus.textContent = 'CAMERA INACTIVE';
    fpsDisplay.textContent = '0 FPS';
    latencyDisplay.textContent = '0 ms';
  }

  toggleCameraBtn.addEventListener('click', () => {
    if (isCameraRunning) stopCamera();
    else startCamera();
  });

  startCameraBtnInline.addEventListener('click', startCamera);

  // Reset Workout Counter
  resetBtn.addEventListener('click', async () => {
    try {
      const resp = await fetch('/reset_counter', { method: 'POST' });
      const data = await resp.json();
      mainRepCount.textContent = '0';
      partialRepCount.textContent = '0';
      hudRepCount.textContent = '0';
      currentPhaseText.textContent = 'REST';
      updateRepRing(0);
      cuesList.innerHTML = `
        <div class="cue-item cue-info">
          <span class="cue-icon">ℹ</span>
          <span class="cue-text">Workout reset. Begin when ready.</span>
        </div>
      `;
    } catch (e) {
      console.error('Reset error:', e);
    }
  });

  // Main Processing Loop
  async function processVideoLoop() {
    if (!isCameraRunning) return;

    const now = performance.now();
    frameCount++;
    if (now - lastFrameTime >= 1000) {
      currentFps = frameCount;
      frameCount = 0;
      lastFrameTime = now;
      fpsDisplay.textContent = `${currentFps} FPS`;
    }

    if (!isProcessingFrame && video.readyState === video.HAVE_ENOUGH_DATA) {
      isProcessingFrame = true;

      // Draw current video frame to capture canvas
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const frameDataUrl = captureCanvas.toDataURL('image/jpeg', 0.65);

      try {
        const resp = await fetch('/predict_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: frameDataUrl })
        });

        if (resp.ok) {
          const result = await resp.json();
          renderAnalysisResult(result);
        }
      } catch (err) {
        console.warn('Frame processing request error:', err);
      } finally {
        isProcessingFrame = false;
      }
    }

    animationFrameId = requestAnimationFrame(processVideoLoop);
  }

  // Render Skeleton & HUD
  function renderAnalysisResult(res) {
    latencyDisplay.textContent = `${res.latency_ms || 0} ms`;

    // Clear overlay canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!res.pose_detected) {
      formStatusBadge.textContent = 'NO POSE DETECTED';
      formStatusBadge.className = 'form-pill form-evaluating';
      exerciseStateBadge.textContent = 'WAITING';
      hudElbowAngle.textContent = '--°';
      return;
    }

    // Render skeleton if enabled
    if (skeletonToggle.checked && res.landmarks) {
      drawSkeletonOverlay(res.landmarks, res.active_elbow_angle);
    }

    // Update Rep Counters & State
    mainRepCount.textContent = res.rep_count;
    hudRepCount.textContent = res.rep_count;
    partialRepCount.textContent = res.partial_rep_count;
    updateRepRing(res.rep_count);

    if (res.rep_completed) {
      playRepChime(true);
      triggerRepFlash();
    } else if (res.partial_completed) {
      playRepChime(false);
    }

    // Exercise Phase Badge
    const state = res.state || 'EXTENDED';
    currentPhaseText.textContent = state;
    exerciseStateBadge.textContent = state;
    exerciseStateBadge.className = `status-pill status-${state.toLowerCase()}`;

    // Angle Readings
    hudElbowAngle.textContent = `${res.active_elbow_angle}°`;
    leftElbowText.textContent = `${res.left_elbow_angle}°`;
    rightElbowText.textContent = `${res.right_elbow_angle}°`;
    torsoTiltText.textContent = `${res.torso_angle || 0}°`;
    shoulderSymText.textContent = `${res.shoulder_symmetry || 0}`;

    // Range of Motion status
    if (res.active_elbow_angle <= 55) {
      romStatusText.textContent = 'Full Contraction';
      romStatusText.className = 'stat-value text-accent';
    } else if (res.active_elbow_angle >= 140) {
      romStatusText.textContent = 'Full Extension';
      romStatusText.className = 'stat-value text-accent';
    } else {
      romStatusText.textContent = 'Curling';
      romStatusText.className = 'stat-value';
    }

    // ML Form Classification
    if (res.ml_prediction) {
      const pred = res.ml_prediction;
      const conf = pred.confidence_pct || 0;
      confidenceTag.textContent = `Confidence: ${conf}%`;

      mlPredictionTitle.textContent = pred.display;

      if (pred.raw === 'Correct') {
        formStatusBadge.textContent = 'CORRECT FORM';
        formStatusBadge.className = 'form-pill form-correct';
        predictionBox.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        mlPredictionDesc.textContent = 'Excellent isolation, steady upper-arm positioning, and balanced trajectory.';
      } else if (pred.raw === 'Partial') {
        formStatusBadge.textContent = 'PARTIAL REP';
        formStatusBadge.className = 'form-pill form-partial';
        predictionBox.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        mlPredictionDesc.textContent = 'Range of motion is cut short. Curl higher at the top and extend completely down.';
      } else {
        formStatusBadge.textContent = 'BODY SWING';
        formStatusBadge.className = 'form-pill form-swing';
        predictionBox.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        mlPredictionDesc.textContent = 'Momentum detected. Avoid swinging torso or pulling elbows off the curl pad.';
      }
    }

    // Probabilities
    if (res.probabilities) {
      const pCorr = Math.round((res.probabilities.Correct || 0) * 100);
      const pPart = Math.round((res.probabilities.Partial || 0) * 100);
      const pSwing = Math.round((res.probabilities.Leg_Drive || 0) * 100);

      probCorrect.style.width = `${pCorr}%`;
      probCorrectText.textContent = `${pCorr}%`;

      probPartial.style.width = `${pPart}%`;
      probPartialText.textContent = `${pPart}%`;

      probSwing.style.width = `${pSwing}%`;
      probSwingText.textContent = `${pSwing}%`;
    }

    // Coaching Cues
    if (res.feedback_cues && res.feedback_cues.length > 0) {
      cuesList.innerHTML = res.feedback_cues.map(cue => `
        <div class="cue-item cue-${cue.type}">
          <span class="cue-icon">${cue.type === 'success' ? '✓' : cue.type === 'warning' ? '⚠' : 'ℹ'}</span>
          <span class="cue-text">${cue.message}</span>
        </div>
      `).join('');
    }
  }

  // Draw Pose Skeleton & Angles directly on canvas
  function drawSkeletonOverlay(landmarks, activeAngle) {
    const lmMap = {};
    landmarks.forEach(l => {
      lmMap[l.id] = {
        x: l.x * canvas.width,
        y: l.y * canvas.height,
        vis: l.visibility
      };
    });

    // Upper body bone pairs (indices match MediaPipe)
    const bones = [
      [11, 12], // Shoulder span
      [11, 13], // Left upper arm
      [13, 15], // Left forearm
      [12, 14], // Right upper arm
      [14, 16], // Right forearm
      [11, 23], // Left torso
      [12, 24], // Right torso
    ];

    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.85)';
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#00f2fe';

    // Draw Bones
    bones.forEach(([p1, p2]) => {
      const a = lmMap[p1];
      const b = lmMap[p2];
      if (a && b && a.vis > 0.4 && b.vis > 0.4) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    });

    ctx.shadowBlur = 0;

    // Draw Joint Landmark Nodes
    landmarks.forEach(lm => {
      if (lm.visibility > 0.4) {
        const x = lm.x * canvas.width;
        const y = lm.y * canvas.height;

        ctx.fillStyle = '#07090e';
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 2.5;

        // Key joints larger (elbows & wrists)
        const radius = (lm.id === 13 || lm.id === 14 || lm.id === 15 || lm.id === 16) ? 7 : 5;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    });

    // Draw Angle Arc at Elbows
    [13, 14].forEach(elbowId => {
      const elbow = lmMap[elbowId];
      if (elbow && elbow.vis > 0.4) {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 13px Outfit, sans-serif';
        ctx.shadowColor = '#000';
        ctx.shadowBlur = 4;
        ctx.fillText(`${Math.round(activeAngle)}°`, elbow.x + 10, elbow.y - 10);
        ctx.shadowBlur = 0;
      }
    });
  }

  // Circular Rep Progress Ring
  function updateRepRing(count) {
    const circumference = 314;
    // Target 10 reps per set
    const progress = Math.min((count % 10) / 10, 1.0);
    const offset = circumference - (progress * circumference);
    repRingProgress.style.strokeDashoffset = offset;
  }

  function triggerRepFlash() {
    mainRepCount.style.transform = 'scale(1.25)';
    mainRepCount.style.color = 'var(--accent-cyan)';
    setTimeout(() => {
      mainRepCount.style.transform = 'scale(1)';
      mainRepCount.style.color = '#fff';
    }, 300);
  }
});
