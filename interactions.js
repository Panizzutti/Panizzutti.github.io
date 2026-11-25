function fitCanvas(canvas, context) {
  const { width, height } = canvas.getBoundingClientRect();
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
  return { width, height };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mix(start, end, progress) {
  return start + (end - start) * progress;
}

function meshProjection(width, height) {
  const scale = Math.min(width / 5.7, height / 3.73);
  return {
    x: x => width / 2 + (x + 1.27) * scale,
    y: y => height / 2 + (y - .08) * scale,
  };
}

function fractionToOutline(x, y, outline, [centerX, centerY]) {
  const dx = x - centerX;
  const dy = y - centerY;
  if (Math.hypot(dx, dy) < 1e-6) return 0;
  let boundary = Infinity;
  for (let index = 0; index < outline.length; index++) {
    const [ax, ay] = outline[index];
    const [bx, by] = outline[(index + 1) % outline.length];
    const ex = bx - ax;
    const ey = by - ay;
    const cross = dx * ey - dy * ex;
    if (Math.abs(cross) < 1e-7) continue;
    const alongRay = ((ax - centerX) * ey - (ay - centerY) * ex) / cross;
    const alongEdge = ((ax - centerX) * dy - (ay - centerY) * dx) / cross;
    if (alongRay > 0 && alongEdge >= -1e-8 && alongEdge <= 1 + 1e-8) boundary = Math.min(boundary, alongRay);
  }
  return Number.isFinite(boundary) ? 1 / boundary : Infinity;
}

function warpStages(baseStages, customSourceOutline) {
  const baseTarget = baseStages.target;
  const baseFrames = baseStages.frames;
  const n = baseTarget.length;
  const finalStar = baseFrames[2].outline;
  const learnedMatches = baseFrames[2].matches;

  let customSource = baseFrames[0].outline;
  let sourceDelta = baseFrames[0].outline.map(() => [0, 0]);
  let origin = [-2.46, 0.435];

  if (customSourceOutline) {
    const xs = customSourceOutline.map(([x]) => x);
    const ys = customSourceOutline.map(([, y]) => y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const scale = 2.35 / Math.max(maxX - minX, maxY - minY);
    customSource = customSourceOutline.map(([x, y]) => [(x - centerX) * scale - 2.46, (y - centerY) * scale + 0.435]);
    sourceDelta = customSource.map(([x, y], i) => [
      x - baseFrames[0].outline[i][0],
      y - baseFrames[0].outline[i][1],
    ]);
    origin = [-centerX * scale - 2.46, 0.435 - centerY * scale];
  }

  // True intermediate RL-regularized SGD step: halfway from source toward finalStar with Laplacian contour smoothing (5 emerging star lobes, never a hexagon)
  const rawMid = customSource.map(([sx, sy], i) => [
    mix(sx, finalStar[i][0], 0.54),
    mix(sy, finalStar[i][1], 0.54),
  ]);
  const midOutline = rawMid.map(([x, y], i) => {
    const prev2 = rawMid[(i - 2 + n) % n];
    const prev1 = rawMid[(i - 1 + n) % n];
    const next1 = rawMid[(i + 1) % n];
    const next2 = rawMid[(i + 2) % n];
    return [
      0.42 * x + 0.22 * (prev1[0] + next1[0]) + 0.07 * (prev2[0] + next2[0]),
      0.42 * y + 0.22 * (prev1[1] + next1[1]) + 0.07 * (prev2[1] + next2[1]),
    ];
  });

  const frames = [
    {
      outline: customSource,
      matches: learnedMatches,
    },
    {
      outline: midOutline,
      matches: learnedMatches,
    },
    {
      outline: finalStar.map(([x, y], i) => [
        x + sourceDelta[i][0] * 0.1,
        y + sourceDelta[i][1] * 0.1,
      ]),
      matches: learnedMatches,
    },
  ];

  return { source: customSource, frames, target: baseTarget, origin };
}

function drawShapeCanvas(mesh) {
  const canvas = document.querySelector(".shape-background");
  const context = canvas.getContext("2d");
  if (!context) return;

  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const samples = [];
  const rings = [];
  const pointCount = mesh?.pointCount ?? 88;
  const angles = Array.from({ length: pointCount }, (_, index) => -index / pointCount * Math.PI * 2);
  const start = angles.map(angle => 1 - .12 * Math.sin(2 * angle - .4) + .08 * Math.cos(3 * angle - .55));
  let current = [...start];
  let goal = start;
  let suggestion;
  let morphSource;
  let hasCustomOutline = false;
  let visibleShape;
  let press;
  let layout;
  let frame = 0;

  function points(radii) {
    return angles.map((angle, index) => [
      Math.cos(angle) * radii[index], .84 * Math.sin(angle) * radii[index],
    ]);
  }

  function fit(examples) {
    return angles.map((angle, index) => {
      let total = 1;
      let prediction = start[index];
      for (const sample of examples) {
        const weight = 24 * Math.exp(-5.5 * (1 - Math.cos(angle - sample.angle)));
        const targetRadius = sample.radius >= 1 ? sample.radius * 1.03 : sample.radius * .97;
        total += weight;
        prediction += weight * targetRadius;
      }
      return prediction / total;
    });
  }

  function updateShape() {
    goal = samples.length ? fit(samples) : start;
    morphSource = mesh?.setOutline(hasCustomOutline ? points(goal) : undefined);
    schedule();
  }

  function measure() {
    const { width, height } = fitCanvas(canvas, context);
    const compact = width < 700;
    layout = {
      width, height,
      cx: width * (compact ? .84 : .78),
      cy: height * (compact ? .56 : .48),
      radius: Math.min(width * (compact ? .23 : .18), height * .32),
    };
    schedule();
  }

  function trace(outline) {
    const first = outline[0];
    const last = outline[outline.length - 1];
    context.beginPath();
    context.moveTo((first[0] + last[0]) / 2, (first[1] + last[1]) / 2);
    for (let index = 0; index < outline.length; index++) {
      const [x, y] = outline[index];
      const [nextX, nextY] = outline[(index + 1) % outline.length];
      context.quadraticCurveTo(x, y, (x + nextX) / 2, (y + nextY) / 2);
    }
    context.closePath();
  }

  function draw() {
    frame = 0;
    const { width, height, cx, cy, radius } = layout;
    let moving = false;
    current = current.map((value, index) => {
      const difference = goal[index] - value;
      if (Math.abs(difference) <= .001 || motion.matches) return goal[index];
      moving = true;
      return value + difference * .23;
    });
    context.clearRect(0, 0, width, height);

    const target = mesh?.transitionTarget();
    const approach = target ? target.approach : 0;
    const departure = target ? target.departure : 0;
    const progress = morphSource && !motion.matches ? approach * (1 - departure) : 0;
    const fade = morphSource && !motion.matches
      ? 1 - clamp((approach - .55) / .45, 0, 1)
      : 1 - approach;
    const opacity = Math.max(fade, departure * .65);
    visibleShape = undefined;
    if (opacity <= .001) return;

    const outline = points(current).map(([x, y], index) => {
      const startX = cx + x * radius;
      const startY = cy + y * radius;
      if (!progress) return [startX, startY];
      return [mix(startX, target.x(morphSource.source[index][0]), progress),
        mix(startY, target.y(morphSource.source[index][1]), progress)];
    });
    const center = progress ? [mix(cx, target.x(morphSource.origin[0]), progress),
      mix(cy, target.y(morphSource.origin[1]), progress)] : [cx, cy];
    if (opacity > .18) visibleShape = { outline, center };
    const shade = context.createLinearGradient(cx - radius, cy - radius * .7, cx + radius, cy + radius);
    shade.addColorStop(0, "rgba(236, 213, 185, .14)");
    shade.addColorStop(.56, "rgba(224, 188, 150, .35)");
    shade.addColorStop(1, "rgba(190, 133, 92, .24)");
    context.globalAlpha = opacity;
    trace(outline);
    context.fillStyle = shade;
    context.fill();
    const ringOpacity = opacity * Math.max(0, 1 - progress * 1.4);
    if (ringOpacity > .001) {
      context.globalAlpha = ringOpacity;
      const sortedRings = [...rings].sort((a, b) => b - a);
      sortedRings.forEach((ring, index) => {
        const isDark = (index + 1) % 5 === 0;
        context.strokeStyle = isDark ? "rgba(140, 78, 43, .68)" : "rgba(150, 88, 51, .32)";
        context.lineWidth = isDark ? 1.35 : .85;
        trace(outline.map(([x, y]) => [mix(center[0], x, ring), mix(center[1], y, ring)]));
        context.stroke();
      });
      context.globalAlpha = opacity;
    }
    trace(outline);
    context.strokeStyle = "rgba(150, 88, 51, .68)";
    context.lineWidth = 1.35;
    context.stroke();

    if (suggestion && progress === 0) {
      const prediction = points(fit([...samples, suggestion])).map(([x, y]) => [cx + x * radius, cy + y * radius]);
      context.strokeStyle = "rgba(128, 75, 44, .53)";
      context.setLineDash([4, 6]);
      trace(prediction);
      context.stroke();
      context.setLineDash([]);
    }
    context.globalAlpha = 1;
    if (moving) schedule();
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(draw);
  }

  function isControl(event) {
    return event.target.closest?.("a, button, input, select, textarea, summary, label, .gdp-explorer, [contenteditable]");
  }

  function readExample(event) {
    const { cx, cy, radius } = layout;
    const x = (event.clientX - cx) / radius;
    const y = (event.clientY - cy) / (radius * .84);
    const distance = Math.hypot(x, y);
    return { angle: Math.atan2(y, x), radius: clamp(distance, .16, 4.5) };
  }

  function radiusAt(event) {
    if (!visibleShape) return Infinity;
    return fractionToOutline(event.clientX, event.clientY, visibleShape.outline, visibleShape.center);
  }

  function addRing(fraction) {
    const preferred = clamp(fraction, .05, .96);
    for (let step = 0; step <= 60; step++) {
      const offset = step === 0 ? 0 : Math.ceil(step / 2) * .016 * (step % 2 ? 1 : -1);
      const ring = preferred + offset;
      if (ring < .05 || ring > .96 || rings.some(existing => Math.abs(existing - ring) < .014)) continue;
      rings.push(ring);
      return;
    }
  }

  let lastPointer;

  function updateSuggestion(event) {
    if (!event) return;
    const canDeform = radiusAt(event) > 1 || event.shiftKey;
    suggestion = !isControl(event) && canDeform ? readExample(event) : undefined;
    schedule();
  }

  document.addEventListener("pointermove", event => {
    if (!event.isPrimary || event.pointerType === "touch") return;
    lastPointer = event;
    updateSuggestion(event);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Shift" && lastPointer) updateSuggestion({ ...lastPointer, shiftKey: true, target: lastPointer.target });
  });
  document.addEventListener("keyup", event => {
    if (event.key === "Shift" && lastPointer) updateSuggestion({ ...lastPointer, shiftKey: false, target: lastPointer.target });
  });
  document.documentElement.addEventListener("pointerleave", () => {
    lastPointer = undefined;
    suggestion = undefined;
    schedule();
  });
  document.addEventListener("pointerdown", event => {
    if (!event.isPrimary || event.button !== 0 || isControl(event)) return;
    if (event.shiftKey) event.preventDefault();
    press = { x: event.clientX, y: event.clientY, id: event.pointerId };
  });
  document.addEventListener("pointercancel", () => { press = undefined; });
  document.addEventListener("pointerup", event => {
    if (press?.id !== event.pointerId) return;
    const travel = Math.hypot(event.clientX - press.x, event.clientY - press.y);
    press = undefined;
    if (travel > 8 || isControl(event)) return;
    if (event.detail > 1 || event.shiftKey) window.getSelection()?.removeAllRanges();
    const inBackground = radiusAt(event);
    if (inBackground <= 1 && !event.shiftKey) {
      addRing(inBackground);
    } else {
      hasCustomOutline = true;
      const sample = readExample(event);
      const redundant = samples.findIndex(previous => Math.cos(previous.angle - sample.angle) > .976);
      if (redundant >= 0) samples.splice(redundant, 1);
      samples.push(sample);
    }
    suggestion = undefined;
    updateShape();
  });
  new ResizeObserver(measure).observe(canvas);
  window.addEventListener("scroll", schedule, { passive: true });
  motion.addEventListener("change", schedule);
  measure();
}

function setupMeshStory(stages) {
  const story = document.getElementById("mesh-story");
  const canvas = story.querySelector(".mesh-art");
  const context = canvas.getContext("2d");
  if (!context) return;

  const { frames: referenceFrames, target: starTarget } = stages;
  const steps = [...story.querySelectorAll(".mesh-step")];
  const caption = story.querySelector(".mesh-frame");
  const compact = window.matchMedia("(max-width: 700px)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const colors = starTarget.map((_, index) => "hsl(" + (265 - index / starTarget.length * 270) + " 86% 63%)");
  let activeTarget = starTarget;
  let customOutline;
  let activeFrames = warpStages(stages).frames;
  let size;
  let scheduled = 0;
  let previous = -1;
  let currentStep = -1;
  let matchAnimProgress = 1;
  let matchAnimStart = 0;
  let pinnedStage = null;

  function rebuildActiveFrames() {
    const warped = warpStages(stages, customOutline);
    activeTarget = warped.target;
    activeFrames = warped.frames;
    return warped;
  }

  function triggerMatchSweep() {
    if (reducedMotion.matches) {
      matchAnimProgress = 1;
      previous = -1;
      schedule();
      return;
    }
    matchAnimProgress = 0;
    matchAnimStart = performance.now();
    previous = -1;
    schedule();
  }

  function draw(now = performance.now()) {
    scheduled = 0;
    if (matchAnimProgress < 1) {
      const elapsed = Math.max(0, now - matchAnimStart);
      const t = Math.min(1, elapsed / 680);
      matchAnimProgress = 1 - (1 - t) ** 3;
      previous = -1;
    }

    const rect = story.getBoundingClientRect();
    const startY = innerHeight * (compact.matches ? 0.82 : 0.76);
    const spanY = innerHeight * (compact.matches ? 0.56 : 0.54);
    const scrollFrac = clamp((startY - rect.top) / Math.max(120, spanY), 0, 1);

    let lineReach;
    let stage;
    let step;
    if (pinnedStage !== null) {
      stage = pinnedStage;
      step = pinnedStage;
      lineReach = matchAnimProgress;
    } else {
      const scrollLineReach = reducedMotion.matches ? 1 : clamp(scrollFrac / 0.26, 0, 1);
      lineReach = Math.min(scrollLineReach, matchAnimProgress);
      if (scrollFrac <= 0.26) {
        stage = 0;
      } else if (scrollFrac <= 0.63) {
        stage = (scrollFrac - 0.26) / 0.37;
      } else {
        stage = 1 + (scrollFrac - 0.63) / 0.37;
      }
      step = stage < 0.45 ? 0 : (stage < 1.45 ? 1 : 2);
    }
    const progress = reducedMotion.matches ? step : stage;
    if (step !== currentStep) {
      steps.forEach((button, index) => {
        if (index === step) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      });
      caption.textContent = steps[step].querySelector("strong").textContent;
      currentStep = step;
    }
    const stateKey = progress * 10 + lineReach;
    if (Math.abs(stateKey - previous) < .0005 && matchAnimProgress >= 1) return;
    previous = stateKey;

    const from = Math.min(1, Math.floor(progress));
    const blend = progress - from;
    const first = activeFrames[from];
    const next = activeFrames[from + 1];
    const { width, height } = size;
    const { x: toX, y: toY } = meshProjection(width, height);
    const outline = first.outline.map(([x, y], index) => [
      x + (next.outline[index][0] - x) * blend,
      y + (next.outline[index][1] - y) * blend,
    ]);
    context.clearRect(0, 0, width, height);
    context.lineWidth = .85;

    if (lineReach > .01) {
      for (let index = 0; index < outline.length; index += 2) {
        const [x, y] = outline[index];
        const sx = toX(x);
        const sy = toY(y);
        context.strokeStyle = colors[index];
        for (const [frame, opacity] of [[first, 1 - blend], [next, blend]]) {
          if (opacity < .02) continue;
          const [mx, my] = frame.matches[index];
          const tx = mix(sx, toX(mx), lineReach);
          const ty = mix(sy, toY(my), lineReach);
          context.globalAlpha = .48 * opacity * Math.min(1, lineReach * 1.25);
          context.beginPath();
          context.moveTo(sx, sy);
          context.lineTo(tx, ty);
          context.stroke();
        }
      }
    }

    context.globalAlpha = 1;
    context.beginPath();
    activeTarget.forEach(([x, y], index) => {
      if (index === 0) context.moveTo(toX(x), toY(y));
      else context.lineTo(toX(x), toY(y));
    });
    context.closePath();
    context.strokeStyle = "#c8b8a8";
    context.lineWidth = 1.55;
    context.setLineDash([4, 3]);
    context.stroke();
    context.setLineDash([]);

    for (let index = 0; index < outline.length; index++) {
      const [x, y] = outline[index];
      const [endX, endY] = outline[(index + 1) % outline.length];
      context.strokeStyle = colors[index];
      context.lineWidth = 1.35;
      context.beginPath();
      context.moveTo(toX(x), toY(y));
      context.lineTo(toX(endX), toY(endY));
      context.stroke();
      context.fillStyle = colors[index];
      context.beginPath();
      context.arc(toX(x), toY(y), width < 380 ? 1.48 : 2.05, 0, 2 * Math.PI);
      context.fill();
    }

    if (matchAnimProgress < 1) schedule();
  }

  function schedule() {
    if (!scheduled) scheduled = requestAnimationFrame(draw);
  }

  function measure() {
    size = fitCanvas(canvas, context);
    previous = -1;
    schedule();
  }

  story.classList.add("is-interactive");
  story.querySelector(".mesh-fallback").hidden = true;
  story.querySelector(".mesh-steps").hidden = false;
  caption.hidden = false;
  canvas.hidden = false;
  new ResizeObserver(measure).observe(canvas);
  measure();
  window.addEventListener("scroll", () => {
    pinnedStage = null;
    schedule();
  }, { passive: true });
  compact.addEventListener("change", schedule);
  reducedMotion.addEventListener("change", schedule);
  steps.forEach((step, idx) => {
    step.addEventListener("click", () => {
      pinnedStage = idx;
      previous = -1;
      if (idx === 0) triggerMatchSweep();
      else {
        matchAnimProgress = 1;
        schedule();
      }
    });
  });
  return {
    pointCount: starTarget.length,
    setOutline(outline) {
      customOutline = outline;
      const sample = rebuildActiveFrames();
      previous = -1;
      schedule();
      return outline ? sample : undefined;
    },
    transitionTarget() {
      const rect = canvas.getBoundingClientRect();
      const projection = meshProjection(rect.width, rect.height);
      return {
        approach: clamp((innerHeight * .95 - rect.top) / (innerHeight * .53), 0, 1),
        departure: clamp(-rect.bottom / (innerHeight * .42), 0, 1),
        x: value => rect.left + projection.x(value),
        y: value => rect.top + projection.y(value),
      };
    },
  };
}

const meshStory = typeof deformationStages === "undefined" ? undefined : setupMeshStory(deformationStages);
drawShapeCanvas(meshStory);
setupGdpoissonExplorer();

function setupGdpoissonExplorer() {
  const root = document.getElementById("gdp-explorer");
  if (!root) return;
  const canvas = document.getElementById("gdp-canvas");
  const context = canvas?.getContext("2d");
  if (!context) return;

  const fallback = root.querySelector(".gdp-fallback");
  const controls = root.querySelector(".gdp-controls");
  const caption = root.querySelector(".gdp-caption");
  if (fallback) fallback.hidden = true;
  if (controls) controls.hidden = false;
  canvas.hidden = false;
  if (caption) caption.hidden = false;

  const lambdaInput = document.getElementById("gdp-lambda");
  const thetaInput = document.getElementById("gdp-theta");
  const lambdaOut = document.getElementById("gdp-lambda-out");
  const thetaOut = document.getElementById("gdp-theta-out");
  const varOut = document.getElementById("gdp-var");
  const ratioOut = document.getElementById("gdp-ratio");
  const codeOut = document.getElementById("gdp-code");
  const hoverOut = document.getElementById("gdp-hover");
  const presets = [...root.querySelectorAll(".gdp-preset")];

  const lanczosCoeffs = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  function lgamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    let x = 0.99999999999980993;
    const t = z - 1;
    for (let i = 0; i < lanczosCoeffs.length; i++) x += lanczosCoeffs[i] / (t + i + 1);
    const w = t + lanczosCoeffs.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (t + 0.5) * Math.log(w) - w + Math.log(x);
  }

  const THETA_MIN = 0.001;
  const THETA_MAX = 3;

  function sliderToTheta(u) {
    const t = clamp(u, 0, 1);
    return THETA_MIN + (THETA_MAX - THETA_MIN) * t * t;
  }

  function thetaToSlider(theta) {
    const clamped = clamp(theta, THETA_MIN, THETA_MAX);
    return Math.sqrt((clamped - THETA_MIN) / (THETA_MAX - THETA_MIN));
  }

  function formatTheta(theta) {
    if (theta < 0.1) return Number(theta.toFixed(3)).toString();
    return theta.toFixed(2);
  }

  if (thetaInput.dataset.theta) {
    thetaInput.value = thetaToSlider(parseFloat(thetaInput.dataset.theta)).toFixed(4);
  }

  function gammaUpperQ(a, x) {
    if (x <= 0) return 1;
    if (a <= 0) return 0;
    if (Math.abs(x - a) > 14 * Math.sqrt(a) + 40) return x < a ? 1 : 0;
    const logPref = a * Math.log(x) - x - lgamma(a);
    if (x < a + 1) {
      let term = 1 / a;
      let sum = term;
      for (let n = 1; n <= 2500; n++) {
        term *= x / (a + n);
        sum += term;
        if (Math.abs(term) < Math.abs(sum) * 1e-14) break;
      }
      return clamp(1 - sum * Math.exp(logPref), 0, 1);
    }
    let b = x + 1 - a;
    let c = 1e30;
    let d = 1 / b;
    let h = d;
    for (let i = 1; i <= 2500; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = b + an / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-14) break;
    }
    return clamp(h * Math.exp(logPref), 0, 1);
  }

  function gdpTerm(m, lambda, theta) {
    if (m <= 0) return 0;
    const a = m / theta;
    const x = lambda / theta;
    if (Math.abs(x - a) > 14 * Math.sqrt(a) + 40) return m > lambda ? m - lambda : 0;
    const qUpper = gammaUpperQ(a, x);
    const dGamma = Math.exp(a * Math.log(x) - x - lgamma(1 + a));
    return (m - lambda) * qUpper + m * dGamma;
  }

  function dgdpois(k, lambda, theta) {
    if (k < 0) return 0;
    if (k === 0) return Math.max(0, gdpTerm(1, lambda, theta));
    if (k === 1) return Math.max(0, gdpTerm(2, lambda, theta) - 2 * gdpTerm(1, lambda, theta));
    return Math.max(0, gdpTerm(k + 1, lambda, theta) - 2 * gdpTerm(k, lambda, theta) + gdpTerm(k - 1, lambda, theta));
  }

  function dpois(k, lambda) {
    if (k < 0) return 0;
    return Math.exp(k * Math.log(lambda) - lambda - lgamma(k + 1));
  }

  let hoveredK = null;
  let barGeometry = [];

  function drawChart() {
    const lambda = parseFloat(lambdaInput.value);
    const theta = sliderToTheta(parseFloat(thetaInput.value));
    const thetaDigits = theta < 0.1 ? 3 : 2;
    const displayTheta = Number(theta.toFixed(thetaDigits));
    lambdaOut.textContent = lambda.toFixed(1);
    thetaOut.textContent = theta < 0.1 ? theta.toFixed(3) : theta.toFixed(2);

    presets.forEach(btn => {
      const targetTheta = parseFloat(btn.dataset.theta);
      const tol = targetTheta < 0.1 ? 0.002 : 0.02;
      btn.setAttribute("aria-pressed", Math.abs(targetTheta - theta) < tol ? "true" : "false");
    });

    const maxK = Math.max(15, Math.min(28, Math.ceil(lambda + 4 * Math.sqrt(lambda * Math.max(1, theta)) + 2)));
    const gdpMass = [];
    const poisMass = [];
    let mean = 0;
    let secondMoment = 0;
    for (let k = 0; k <= 45; k++) {
      const p = dgdpois(k, lambda, theta);
      mean += k * p;
      secondMoment += k * k * p;
      if (k <= maxK) {
        gdpMass.push(p);
        poisMass.push(dpois(k, lambda));
      }
    }
    const variance = Math.max(0, secondMoment - mean * mean);
    varOut.textContent = variance.toFixed(2);
    ratioOut.textContent = (variance / lambda).toFixed(2);
    codeOut.textContent = `dgdpois(0:${maxK}, lambda = ${Number(lambda.toFixed(1))}, theta = ${displayTheta})`;

    const { width, height } = fitCanvas(canvas, context);
    context.clearRect(0, 0, width, height);

    const padLeft = 44;
    const padRight = 14;
    const padTop = 16;
    const padBottom = 30;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;
    const maxY = Math.max(0.22, ...gdpMass, ...poisMass) * 1.1;

    context.strokeStyle = "#dfd3c2";
    context.fillStyle = "#605247";
    context.font = '500 11px "Space Grotesk", -apple-system, sans-serif';
    context.textAlign = "right";
    context.textBaseline = "middle";
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const val = (maxY / ticks) * i;
      const y = padTop + plotH - (val / maxY) * plotH;
      context.beginPath();
      context.moveTo(padLeft, y);
      context.lineTo(width - padRight, y);
      context.stroke();
      context.fillText(val.toFixed(2), padLeft - 7, y);
    }

    const count = maxK + 1;
    const slotW = plotW / count;
    const barW = Math.min(24, Math.max(6, slotW * 0.56));
    barGeometry = [];

    for (let k = 0; k <= maxK; k++) {
      const cx = padLeft + (k + 0.5) * slotW;
      const gdpH = (gdpMass[k] / maxY) * plotH;
      const gdpY = padTop + plotH - gdpH;
      const isHovered = hoveredK === k;

      context.fillStyle = isHovered ? "#493020" : "#70452e";
      context.beginPath();
      context.roundRect(cx - barW / 2, gdpY, barW, Math.max(1.5, gdpH), [3, 3, 0, 0]);
      context.fill();

      barGeometry.push({ k, left: cx - slotW / 2, right: cx + slotW / 2, cx, gdp: gdpMass[k], pois: poisMass[k] });

      if (maxK <= 18 || k % 2 === 0) {
        context.fillStyle = isHovered ? "#282019" : "#605247";
        context.textAlign = "center";
        context.textBaseline = "top";
        context.fillText(String(k), cx, padTop + plotH + 8);
      }
    }

    context.strokeStyle = "#2b6150";
    context.lineWidth = 1.75;
    context.setLineDash([4, 3]);
    context.beginPath();
    for (let k = 0; k <= maxK; k++) {
      const cx = padLeft + (k + 0.5) * slotW;
      const poisY = padTop + plotH - (poisMass[k] / maxY) * plotH;
      if (k === 0) context.moveTo(cx, poisY);
      else context.lineTo(cx, poisY);
    }
    context.stroke();
    context.setLineDash([]);

    for (let k = 0; k <= maxK; k++) {
      const cx = padLeft + (k + 0.5) * slotW;
      const poisY = padTop + plotH - (poisMass[k] / maxY) * plotH;
      context.fillStyle = hoveredK === k ? "#1b4437" : "#2b6150";
      context.beginPath();
      context.arc(cx, poisY, hoveredK === k ? 4 : 2.8, 0, Math.PI * 2);
      context.fill();
    }

    if (hoveredK !== null && barGeometry[hoveredK]) {
      const b = barGeometry[hoveredK];
      hoverOut.textContent = `k = ${b.k} · Generalized: ${b.gdp.toFixed(4)} · Poisson: ${b.pois.toFixed(4)}`;
    } else {
      hoverOut.textContent = "Hover or tap a bar to compare P(Y = k)";
    }
  }

  lambdaInput.addEventListener("input", drawChart);
  thetaInput.addEventListener("input", drawChart);
  presets.forEach(btn => {
    btn.addEventListener("click", () => {
      thetaInput.value = thetaToSlider(parseFloat(btn.dataset.theta)).toFixed(4);
      drawChart();
    });
  });

  canvas.addEventListener("pointermove", event => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const hit = barGeometry.find(b => x >= b.left && x <= b.right);
    const nextK = hit ? hit.k : null;
    if (nextK !== hoveredK) {
      hoveredK = nextK;
      drawChart();
    }
  });
  canvas.addEventListener("pointerleave", () => {
    if (hoveredK !== null) {
      hoveredK = null;
      drawChart();
    }
  });
  new ResizeObserver(drawChart).observe(canvas);
  drawChart();
}
