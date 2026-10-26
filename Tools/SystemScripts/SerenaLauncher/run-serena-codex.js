const { spawn } = require("node:child_process");
const fs = require("node:fs");

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
		`Serena binary not found. Expected one of: ${candidates.join(", ")}`,
	);
	process.exit(1);
}

// MCP messages are bounded independently of stream chunk sizes. Eight MiB allows
// substantial tool responses; headers are capped at 16 KiB. At most two maximum
// messages may be queued to a backpressured destination before failing closed.
const MAX_MESSAGE_BYTES = 8 * 1024 * 1024;
const MAX_HEADER_BYTES = 16 * 1024;
const MAX_QUEUED_BYTES = 2 * MAX_MESSAGE_BYTES + MAX_HEADER_BYTES;
const FAILURE_KILL_DELAY_MS = 1000;
const { TextDecoder } = require("node:util");
const {
	constants: { signals },
} = require("node:os");

function framePayload(payload) {
	return Buffer.concat([
		Buffer.from(`Content-Length: ${payload.length}\r\n\r\n`, "ascii"),
		payload,
	]);
}

function isWhitespace(value) {
	return value === 0x20 || value === 0x09 || value === 0x0d || value === 0x0a;
}

// Called only after JSON validation. Remove insignificant ASCII whitespace while
// retaining every string byte, escape sequence and numeric lexeme verbatim.
function compactJsonWhitespace(payload) {
	const result = Buffer.allocUnsafe(payload.length);
	let written = 0;
	let quoted = false;
	let escaped = false;
	for (const value of payload) {
		if (quoted) {
			result[written++] = value;
			if (escaped) escaped = false;
			else if (value === 0x5c) escaped = true;
			else if (value === 0x22) quoted = false;
		} else if (value === 0x22) {
			quoted = true;
			result[written++] = value;
		} else if (!isWhitespace(value)) {
			result[written++] = value;
		}
	}
	return result.subarray(0, written);
}

class MessageDecoder {
	constructor(onMessage) {
		this.onMessage = onMessage;
		this.buffer = Buffer.alloc(0);
		this.mode = null;
		this.header = null;
		this.bodyLength = null;
		this.ended = false;
		this.failed = false;
		this.utf8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
	}

	write(chunk) {
		if (this.ended || this.failed)
			throw new Error("Data after decoder completion");
		if (!Buffer.isBuffer(chunk)) chunk = Buffer.from(chunk);
		try {
			let offset = 0;
			while (offset < chunk.length) {
				const capacity =
					MAX_MESSAGE_BYTES + MAX_HEADER_BYTES + 4 - this.buffer.length;
				if (capacity <= 0) throw new Error("Message buffer limit exceeded");
				const size = Math.min(capacity, chunk.length - offset);
				this.buffer = Buffer.concat([
					this.buffer,
					chunk.subarray(offset, offset + size),
				]);
				offset += size;
				this.drain(false);
			}
		} catch (error) {
			this.failed = true;
			this.buffer = Buffer.alloc(0);
			throw error;
		}
	}

	end() {
		if (this.ended) return;
		this.ended = true;
		if (this.failed) return;
		try {
			this.drain(true);
		} catch (error) {
			this.failed = true;
			this.buffer = Buffer.alloc(0);
			throw error;
		}
	}

	validate(payload) {
		if (payload.length > MAX_MESSAGE_BYTES)
			throw new Error("Message exceeds 8 MiB limit");
		// Parse only to validate complete UTF-8 JSON. Never serialize the parsed value:
		// numeric lexemes, whitespace and Unicode bytes must survive unchanged.
		let text;
		try {
			text = this.utf8.decode(payload);
		} catch (error) {
			throw new Error("Invalid UTF-8 message", { cause: error });
		}
		try {
			JSON.parse(text);
		} catch (error) {
			// Engine syntax diagnostics can include payload fragments or secrets.
			throw new Error("Malformed JSON message", { cause: error });
		}
	}

	drain(eof) {
		if (!this.mode) {
			const first = this.buffer.findIndex((value) => !isWhitespace(value));
			if (first < 0) {
				if (this.buffer.length > MAX_MESSAGE_BYTES)
					throw new Error("Message exceeds 8 MiB limit");
				if (eof) this.buffer = Buffer.alloc(0);
				return;
			}
			const initial = String.fromCharCode(this.buffer[first]);
			this.mode = '{["-0123456789tfn'.includes(initial)
				? "json-lines"
				: "framed";
		}
		if (this.mode === "json-lines") {
			for (;;) {
				const newline = this.buffer.indexOf(0x0a);
				if (newline < 0 && !eof) {
					if (this.buffer.length > MAX_MESSAGE_BYTES)
						throw new Error("Message exceeds 8 MiB limit");
					return;
				}
				if (newline < 0 && !this.buffer.length) return;
				const end = newline < 0 ? this.buffer.length : newline;
				let payload = this.buffer.subarray(0, end);
				this.buffer = this.buffer.subarray(newline < 0 ? end : end + 1);
				if (newline >= 0 && payload.at(-1) === 0x0d)
					payload = payload.subarray(0, -1);
				if (payload.every(isWhitespace)) continue;
				this.validate(payload);
				this.onMessage(payload, null);
			}
		}
		for (;;) {
			if (this.bodyLength === null) {
				const headerEnd = this.buffer.indexOf("\r\n\r\n");
				if (headerEnd < 0) {
					if (this.buffer.length > MAX_HEADER_BYTES)
						throw new Error("Header exceeds 16 KiB limit");
					if (eof && this.buffer.length)
						throw new Error("Truncated framed header at EOF");
					return;
				}
				if (headerEnd + 4 > MAX_HEADER_BYTES)
					throw new Error("Header exceeds 16 KiB limit");
				this.header = this.buffer.subarray(0, headerEnd + 4);
				if (
					!this.header.every(
						(value) =>
							(value >= 0x20 && value <= 0x7e) ||
							value === 9 ||
							value === 13 ||
							value === 10,
					)
				) {
					throw new Error("Invalid non-ASCII/control byte in frame header");
				}
				let length = null;
				for (const line of this.buffer
					.subarray(0, headerEnd)
					.toString("ascii")
					.split("\r\n")) {
					const field = /^([!#$%&'*+.^_`|~0-9A-Za-z-]+):[ \t]*(.*)$/.exec(line);
					if (!field) throw new Error("Malformed framed header");
					if (field[1].toLowerCase() === "content-length") {
						if (length !== null || !/^[0-9]+$/.test(field[2].trim()))
							throw new Error("Invalid or duplicate Content-Length");
						length = Number(field[2].trim());
					}
				}
				if (
					length === null ||
					!Number.isSafeInteger(length) ||
					length < 0 ||
					length > MAX_MESSAGE_BYTES
				) {
					throw new Error("Missing or oversized Content-Length");
				}
				this.bodyLength = length;
				this.buffer = this.buffer.subarray(headerEnd + 4);
			}
			if (this.buffer.length < this.bodyLength) {
				if (eof) throw new Error("Truncated framed body at EOF");
				return;
			}
			const payload = this.buffer.subarray(0, this.bodyLength);
			this.validate(payload);
			const framed = Buffer.concat([this.header, payload]);
			this.buffer = this.buffer.subarray(this.bodyLength);
			this.header = null;
			this.bodyLength = null;
			this.onMessage(payload, framed);
		}
	}
}

function runTransport(child, input, output, errorOutput, complete) {
	let failure = null;
	let closed = false;
	let killTimer = null;
	let pendingWrites = 0;
	let completed = false;
	let childCode = null;
	let childSignal = null;
	const errors = [];
	function finish() {
		if (!closed || pendingWrites || completed) return;
		completed = true;
		complete(
			failure
				? 1
				: childSignal
					? 128 + (signals[childSignal] || 1)
					: (childCode ?? 1),
			errors.slice(),
		);
	}
	function diagnostic(message) {
		try {
			errorOutput.write(message);
		} catch (error) {
			// Retain sink failures even when that sink cannot display its own error.
			errors.push(error);
		}
	}
	function terminate(signal) {
		try {
			child.kill(signal);
		} catch (error) {
			errors.push(error);
			diagnostic(`Child termination failed: ${error.message}\n`);
		}
	}
	function fail(error) {
		errors.push(error);
		if (failure) return;
		failure = error;
		input.pause();
		// Consume remaining child bytes after a protocol failure so close can drain.
		child.stdout.resume();
		child.stderr.resume();
		diagnostic(`Serena transport failed: ${error.message}\n`);
		if (!closed) {
			child.stdin.destroy();
			terminate();
			killTimer = setTimeout(() => {
				if (!closed) terminate("SIGKILL");
			}, FAILURE_KILL_DELAY_MS);
			killTimer.unref();
		}
	}
	function forward(target, source, bytes) {
		if (target.destroyed || target.writableEnded)
			throw new Error("Destination closed before message delivery");
		if (target.writableLength + bytes.length > MAX_QUEUED_BYTES)
			throw new Error("Backpressure queue limit exceeded");
		pendingWrites++;
		try {
			if (
				!target.write(bytes, (error) => {
					pendingWrites--;
					if (error) fail(error);
					finish();
				})
			)
				source.pause();
		} catch (error) {
			pendingWrites--;
			throw error;
		}
	}
	const incoming = new MessageDecoder((payload, framed) => {
		if (closed)
			throw new Error("Child closed before pending input message delivery");
		forward(child.stdin, input, framed || framePayload(payload));
	});
	const outgoing = new MessageDecoder((payload, framed) =>
		forward(
			output,
			child.stdout,
			Buffer.concat([
				framed ? compactJsonWhitespace(payload) : payload,
				Buffer.from("\n"),
			]),
		),
	);
	input.on("data", (chunk) => {
		if (!failure && !closed) {
			try {
				incoming.write(chunk);
			} catch (error) {
				fail(error);
			}
		}
	});
	input.on("end", () => {
		if (!failure && !closed) {
			try {
				incoming.end();
				child.stdin.end();
			} catch (error) {
				fail(error);
			}
		}
	});
	child.stdout.on("data", (chunk) => {
		if (!failure) {
			try {
				outgoing.write(chunk);
			} catch (error) {
				fail(error);
			}
		}
	});
	child.stdout.on("end", () => {
		if (!failure) {
			try {
				outgoing.end();
			} catch (error) {
				fail(error);
			}
		}
	});
	child.stderr.on("data", (chunk) => {
		if (!failure) {
			try {
				forward(errorOutput, child.stderr, chunk);
			} catch (error) {
				fail(error);
			}
		}
	});
	child.stdin.on("drain", () => {
		if (!failure && !closed) input.resume();
	});
	output.on("drain", () => {
		if (!failure && !closed) child.stdout.resume();
	});
	errorOutput.on("drain", () => {
		if (!failure && !closed) child.stderr.resume();
	});
	for (const stream of [
		input,
		child.stdin,
		child.stdout,
		child.stderr,
		output,
		errorOutput,
	])
		stream.on("error", fail);
	child.on("error", fail);
	child.on("exit", () => input.pause());
	// 'exit' precedes stdio closure. Only 'close' admits final EOF and completion.
	child.on("close", (code, signal) => {
		if (closed) return;
		closed = true;
		childCode = code;
		childSignal = signal;
		input.pause();
		if (killTimer) clearTimeout(killTimer);
		if (!failure) {
			try {
				incoming.end();
				outgoing.end();
			} catch (error) {
				fail(error);
			}
		}
		finish();
	});
	return {
		get errors() {
			return errors.slice();
		},
		get closed() {
			return closed;
		},
	};
}

function main() {
	const binary = resolveBinary();
	const requestedArgs = process.argv.slice(2);
	const hasProjectArg = requestedArgs.some(
		(arg) =>
			arg === "--project" || arg === "-P" || arg.startsWith("--project="),
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
	runTransport(child, process.stdin, process.stdout, process.stderr, (code) => {
		// Do not force process.exit: pending stdout/stderr writes must drain too.
		process.exitCode = code;
	});
}

module.exports = { MessageDecoder, framePayload, runTransport, main };
if (require.main === module) main();
