const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

function resolveBinary() {
  const candidates = [
    "T:\\projects\\serena-source\\target\\release\\serena.exe",
    "T:\\projects\\serena-source\\target\\debug\\serena.exe",
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  console.error(
    `Serena binary not found. Expected one of: ${candidates.join(", ")}`
  );
  process.exit(1);
}

const binary = resolveBinary();
const requestedArgs = process.argv.slice(2);
const hasProjectArg = requestedArgs.some(
  (arg) => arg === "--project" || arg === "-P" || arg.startsWith("--project=")
);
const projectArgs = hasProjectArg
  ? requestedArgs
  : [...requestedArgs, "--project", process.cwd()];

const child = spawn(binary, projectArgs, {
  cwd: process.cwd(),
  env: { ...process.env },
  stdio: ["pipe", "pipe", "pipe"],
  windowsHide: true,
});

let stdinBuffer = Buffer.alloc(0);
let stdoutBuffer = Buffer.alloc(0);

function writeFramedPayload(payloadText) {
  const payload = Buffer.from(payloadText, "utf8");
  const header = Buffer.from(
    `Content-Length: ${payload.length}\r\n\r\n`,
    "utf8"
  );
  child.stdin.write(Buffer.concat([header, payload]));
}

function tryForwardFramedInput() {
  for (;;) {
    const headerEnd = stdinBuffer.indexOf("\r\n\r\n");
    if (headerEnd === -1) {
      return false;
    }

    const header = stdinBuffer.slice(0, headerEnd).toString("utf8");
    const match = header.match(/Content-Length:\s*(\d+)/i);
    if (!match) {
      return false;
    }

    const contentLength = Number(match[1]);
    const messageStart = headerEnd + 4;
    const messageEnd = messageStart + contentLength;
    if (stdinBuffer.length < messageEnd) {
      return true;
    }

    child.stdin.write(stdinBuffer.slice(0, messageEnd));
    stdinBuffer = stdinBuffer.slice(messageEnd);
  }
}

function tryForwardJsonLines(flushRemainder = false) {
  let forwarded = false;

  for (;;) {
    const newlineIndex = stdinBuffer.indexOf(0x0a);
    if (newlineIndex === -1) {
      break;
    }

    const rawLine = stdinBuffer.slice(0, newlineIndex).toString("utf8");
    stdinBuffer = stdinBuffer.slice(newlineIndex + 1);
    const line = rawLine.replace(/\r$/, "").trim();
    if (!line) {
      continue;
    }
    writeFramedPayload(line);
    forwarded = true;
  }

  if (flushRemainder) {
    const tail = stdinBuffer.toString("utf8").trim();
    if (tail) {
      writeFramedPayload(tail);
      forwarded = true;
    }
    stdinBuffer = Buffer.alloc(0);
  }

  return forwarded;
}

process.stdin.on("data", (chunk) => {
  stdinBuffer = Buffer.concat([stdinBuffer, chunk]);

  if (!tryForwardFramedInput()) {
    tryForwardJsonLines(false);
  }
});

process.stdin.on("end", () => {
  if (!tryForwardFramedInput()) {
    tryForwardJsonLines(true);
  }
  child.stdin.end();
});

function flushPlainJsonIfPresent() {
  const text = stdoutBuffer.toString("utf8").trim();
  if (!text) {
    return false;
  }

  if (text.startsWith("{") || text.startsWith("[")) {
    process.stdout.write(text);
    if (!text.endsWith("\n")) {
      process.stdout.write("\n");
    }
    stdoutBuffer = Buffer.alloc(0);
    return true;
  }

  return false;
}

child.stdout.on("data", (chunk) => {
  stdoutBuffer = Buffer.concat([stdoutBuffer, chunk]);

  for (;;) {
    const headerEnd = stdoutBuffer.indexOf("\r\n\r\n");
    if (headerEnd === -1) {
      break;
    }

    const header = stdoutBuffer.slice(0, headerEnd).toString("utf8");
    const match = header.match(/Content-Length:\s*(\d+)/i);
    if (!match) {
      break;
    }

    const contentLength = Number(match[1]);
    const messageStart = headerEnd + 4;
    const messageEnd = messageStart + contentLength;
    if (stdoutBuffer.length < messageEnd) {
      break;
    }

    const payload = stdoutBuffer.slice(messageStart, messageEnd).toString("utf8");
    process.stdout.write(payload);
    process.stdout.write("\n");
    stdoutBuffer = stdoutBuffer.slice(messageEnd);
  }

  flushPlainJsonIfPresent();
});

child.stderr.on("data", (chunk) => {
  process.stderr.write(chunk);
});

child.on("error", (err) => {
  console.error(`Failed to start Serena wrapper child: ${err.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  flushPlainJsonIfPresent();
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
