/*
 * The page's log: everything that goes wrong (errors, rejected promises, console errors and
 * warnings, scripts that fail to load, and whatever the game reports: its startup steps, the
 * GPU it got, GPU errors, a lost GPU device), with the browser and screen it ran on. A Log
 * button opens it with Copy and Share buttons, so a problem on a phone can be sent as text.
 *
 * A plain script loaded before the game's own code, so it also catches errors that stop the
 * game from loading at all. The last few visits' logs are kept in localStorage too: a GPU crash
 * can leave the page needing a reload. The game talks to it through window.blockLog.
 */
(function () {
  'use strict';
  var started = Date.now();
  var STORE_KEY = 'block.log';
  var MAX_ENTRIES = 300;
  var MAX_DETAIL = 6000;
  var KEPT_VISITS = 3;
  var entries = [];
  var byKey = {};
  var dropped = 0;
  var original = { error: console.error, warn: console.warn };
  var previousVisits = [];
  try {
    previousVisits = JSON.parse(localStorage.getItem(STORE_KEY) || '[]');
    if (!Array.isArray(previousVisits)) previousVisits = [];
  } catch (e) {
    previousVisits = [];
  }

  function seconds(ms) { return '+' + (ms / 1000).toFixed(2) + 's'; }
  function clip(s, n) { s = String(s); return s.length > n ? s.slice(0, n) + '…' : s; }

  /** A thrown value as [one-line text, detail (the stack)]. */
  function describe(value) {
    if (value && typeof value === 'object' && ('message' in value || 'stack' in value)) {
      var name = value.name || (value.constructor && value.constructor.name) || 'Error';
      var text = name + ': ' + value.message;
      var stack = value.stack ? String(value.stack) : '';
      // Chrome's stack repeats the message on its first line.
      if (stack.indexOf(text) === 0) stack = stack.slice(text.length).replace(/^\n/, '');
      return [text, stack];
    }
    if (typeof value === 'string') return [value, ''];
    try { return [clip(JSON.stringify(value), 500), '']; } catch (e) { return [String(value), '']; }
  }

  function add(level, text, detail) {
    text = String(text);
    detail = detail ? clip(detail, MAX_DETAIL) : '';
    var now = Date.now() - started;
    var key = level + '\n' + text + '\n' + detail;
    var entry = byKey[key];
    if (entry) {
      entry.count++;
      entry.last = now;
    } else if (entries.length >= MAX_ENTRIES) {
      dropped++;
    } else {
      entry = { at: now, last: now, level: level, text: text, detail: detail, count: 1 };
      byKey[key] = entry;
      entries.push(entry);
    }
    changed();
  }

  // Uncaught errors, and (in the capture phase) scripts, styles and images that fail to load.
  window.addEventListener('error', function (e) {
    if (e instanceof ErrorEvent) {
      var d = describe(e.error !== undefined && e.error !== null ? e.error : e.message);
      var where = e.filename ? e.filename + ':' + e.lineno + ':' + e.colno : '';
      add('error', 'Uncaught ' + d[0], d[1] || where);
    } else if (e.target && e.target !== window) {
      add('error', 'Failed to load ' + (e.target.src || e.target.href || e.target.tagName));
    }
  }, true);
  window.addEventListener('unhandledrejection', function (e) {
    var d = describe(e.reason);
    add('error', 'Unhandled rejection: ' + d[0], d[1]);
  });
  // The game's (and TensorFlow.js's) console errors and warnings.
  ['error', 'warn'].forEach(function (level) {
    console[level] = function () {
      try {
        var parts = [], details = [];
        for (var i = 0; i < arguments.length; i++) {
          var d = describe(arguments[i]);
          parts.push(d[0]);
          if (d[1]) details.push(d[1]);
        }
        add(level, parts.join(' '), details.join('\n'));
      } catch (err) { /* never let logging break the page */ }
      return original[level].apply(console, arguments);
    };
  });

  function header() {
    var lines = [
      'Block log, page opened ' + new Date(started).toISOString(),
      'Page: ' + location.href,
      'Browser: ' + navigator.userAgent,
      'Screen: ' + innerWidth + 'x' + innerHeight + ' CSS px, devicePixelRatio ' + (window.devicePixelRatio || 1) +
        ', touch points ' + (navigator.maxTouchPoints || 0),
      'WebGPU API: ' + ('gpu' in navigator ? 'present' : 'missing (navigator.gpu is undefined)'),
    ];
    return lines.join('\n');
  }

  function entryText(e) {
    var label = e.level === 'error' ? 'ERROR' : e.level === 'warn' ? 'warn ' : 'info ';
    var line = seconds(e.at) + '  ' + label + '  ' + e.text;
    if (e.count > 1) line += '  (x' + e.count + ', last ' + seconds(e.last) + ')';
    if (e.detail) line += '\n' + e.detail.split('\n').map(function (l) { return '        ' + l.trim(); }).join('\n');
    return line;
  }

  /** This visit's log. */
  function visitText() {
    var out = [header(), ''].concat(entries.map(entryText));
    if (dropped) out.push('(' + dropped + ' more entries not kept)');
    return out.join('\n');
  }

  /** What the page shows right now (the game's status line, error box and HUD). */
  function screenText() {
    var out = [];
    ['status', 'error', 'hud'].forEach(function (id) {
      var el = document.getElementById(id);
      var text = el && el.textContent ? el.textContent.trim() : '';
      if (text) out.push(id + ': ' + text.split('\n').join('\n    '));
    });
    return out.length ? 'On screen when copied:\n' + out.join('\n') : '';
  }

  /** Everything: this visit, what's on screen, and the visits before. */
  function fullText() {
    var out = [visitText()];
    var screen = screenText();
    if (screen) out.push(screen);
    previousVisits.forEach(function (v, i) {
      out.push('=== Previous visit ' + (i + 1) + ' (' + v.started + ') ===\n' + v.text);
    });
    return out.join('\n\n');
  }

  function save() {
    try {
      var visits = [{ started: new Date(started).toISOString(), text: visitText() }].concat(previousVisits.slice(0, KEPT_VISITS - 1));
      localStorage.setItem(STORE_KEY, JSON.stringify(visits));
    } catch (e) { /* storage full or blocked: the log still works on the page */ }
  }

  function counts() {
    var errors = 0, warnings = 0;
    entries.forEach(function (e) { if (e.level === 'error') errors++; else if (e.level === 'warn') warnings++; });
    return { errors: errors, warnings: warnings };
  }

  // ----- UI: a Log button, and a panel with the text and Copy / Share / Close -----

  var button = null, panel = null, textarea = null, note = null;
  var STYLE = [
    // Top right on the start screen; top centre while playing (the touch toolbar has the right edge).
    '.blog-button{position:fixed;z-index:1000;top:calc(8px + env(safe-area-inset-top));right:calc(8px + env(safe-area-inset-right));',
    'padding:5px 12px;border-radius:14px;border:1px solid rgba(255,255,255,.45);background:rgba(0,0,0,.55);color:#fff;',
    'font:12px ui-monospace,Menlo,Consolas,monospace;cursor:pointer;touch-action:manipulation}',
    'body.playing .blog-button{right:auto;left:50%;transform:translateX(-50%)}',
    '.blog-button.warn{background:rgba(140,90,0,.85)}',
    '.blog-button.error{background:rgba(170,30,30,.9);border-color:#fff}',
    '.blog-button[hidden]{display:none}',
    '.blog-panel{position:fixed;z-index:1001;inset:0;display:flex;flex-direction:column;gap:8px;box-sizing:border-box;',
    'padding:calc(10px + env(safe-area-inset-top)) calc(10px + env(safe-area-inset-right)) calc(10px + env(safe-area-inset-bottom)) calc(10px + env(safe-area-inset-left));',
    'background:#0a0c10;color:#e8eaee;font:13px ui-monospace,Menlo,Consolas,monospace}',
    '.blog-panel[hidden]{display:none}',
    '.blog-bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px}',
    '.blog-bar b{font-size:15px;margin-right:auto}',
    '.blog-bar button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid #3b82f6;background:#2563eb;color:#fff;touch-action:manipulation}',
    '.blog-bar button.plain{background:transparent;border-color:#4a5263;color:#e8eaee}',
    '.blog-note{width:100%;color:#9aa3b2;font-size:12px}',
    '.blog-panel textarea{flex:1;min-height:0;width:100%;box-sizing:border-box;resize:none;padding:8px;border-radius:8px;',
    'border:1px solid #2c3340;background:#0b0d11;color:#e8eaee;font:11px/1.45 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}',
  ].join('');

  function stop(e) { e.stopPropagation(); }
  function guard(el) {
    ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'touchstart', 'touchend', 'click', 'contextmenu', 'keydown', 'keyup']
      .forEach(function (type) { el.addEventListener(type, stop); });
  }

  function buildUi() {
    var style = document.createElement('style');
    style.textContent = STYLE;
    document.head.appendChild(style);
    // The Log button: in the game only with ?log in the URL (the panel still opens itself when the
    // game fails); always on the other pages (the benchmark).
    var game = !/(bench|log)\.html$/.test(location.pathname);
    if (game && !/[?&]log(=|&|$)/.test(location.search)) return;
    button = document.createElement('button');
    button.type = 'button';
    button.className = 'blog-button';
    guard(button);
    button.addEventListener('click', open);
    document.body.appendChild(button);
    if (window.MutationObserver) new MutationObserver(updateButton).observe(document.body, { attributes: true, attributeFilter: ['class'] });
    updateButton();
  }

  function updateButton() {
    if (!button) return;
    var c = counts();
    button.className = 'blog-button' + (c.errors ? ' error' : c.warnings ? ' warn' : '');
    button.textContent = c.errors ? 'Log · ' + c.errors + ' error' + (c.errors > 1 ? 's' : '')
      : c.warnings ? 'Log · ' + c.warnings + ' warning' + (c.warnings > 1 ? 's' : '') : 'Log';
  }

  function buildPanel() {
    panel = document.createElement('div');
    panel.className = 'blog-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Log');
    var bar = document.createElement('div');
    bar.className = 'blog-bar';
    var title = document.createElement('b');
    title.textContent = 'Log';
    bar.appendChild(title);
    var actions = [['Copy', copy, '']];
    if (navigator.share) actions.push(['Share', share, '']);
    actions.push(['Close', close, 'plain']);
    actions.forEach(function (a) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = a[0];
      b.className = a[2];
      b.addEventListener('click', a[1]);
      bar.appendChild(b);
    });
    note = document.createElement('div');
    note.className = 'blog-note';
    note.textContent = 'Copy it (or Share it) and paste it into the chat.';
    bar.appendChild(note);
    textarea = document.createElement('textarea');
    textarea.readOnly = true;
    textarea.spellcheck = false;
    textarea.setAttribute('autocapitalize', 'off');
    panel.appendChild(bar);
    panel.appendChild(textarea);
    guard(panel);
    panel.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
    document.body.appendChild(panel);
  }

  function open() {
    if (!document.body) return;
    if (!panel) buildPanel();
    if (document.exitPointerLock && document.pointerLockElement) document.exitPointerLock();
    textarea.value = fullText();
    panel.hidden = false;
  }
  function close() { if (panel) panel.hidden = true; }
  function say(text) { if (note) note.textContent = text; }

  function copy() {
    var text = fullText();
    textarea.value = text;
    function fallback() {
      // Select the text and use the old copy command (works where the clipboard API doesn't).
      var ok = false;
      try {
        textarea.readOnly = false;
        textarea.focus();
        textarea.select();
        textarea.setSelectionRange(0, text.length);
        ok = document.execCommand('copy');
      } catch (e) { ok = false; }
      textarea.readOnly = true;
      say(ok ? 'Copied. Paste it into the chat.' : "Couldn't copy automatically: the text is selected, copy it from there.");
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { say('Copied. Paste it into the chat.'); }, fallback);
    } else {
      fallback();
    }
  }

  function share() {
    navigator.share({ title: 'Block log', text: fullText() }).catch(function (e) {
      if (!e || e.name !== 'AbortError') say("Couldn't share (" + (e && e.message) + '): use Copy instead.');
    });
  }

  function changed() {
    // Saved straight away: if the page crashes (a GPU crash can take the whole tab down), the log
    // up to that moment is still there for log.html.
    save();
    updateButton();
    if (panel && !panel.hidden) textarea.value = fullText();
  }
  window.addEventListener('pagehide', save);

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', buildUi);
  else buildUi();

  window.blockLog = {
    info: function (text, detail) { add('info', text, detail); },
    warn: function (text, detail) { add('warn', text, detail); original.warn.call(console, text, detail || ''); },
    error: function (text, detail) { add('error', text, detail); original.error.call(console, text, detail || ''); },
    /** A thrown value as [text, detail]. */
    describe: describe,
    open: open,
    text: fullText,
  };
})();
