const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { EventEmitter, once } = require("node:events");
const fs = require("node:fs");
const path = require("node:path");
const { PassThrough, Writable } = require("node:stream");
const test = require("node:test");
const vm = require("node:vm");

const sourcePath =
	process.env.SERENA_SOURCE_PATH ||
	path.resolve(
		__dirname,
		"../../Tools/SystemScripts/SerenaLauncher/run-serena-codex.js",
	);
const source = fs.readFileSync(sourcePath, "utf8");
const tick = () => new Promise((resolve) => setImmediate(resolve));

function frame(payload, extra = "") {
	return Buffer.concat([
		Buffer.from(`Content-Length: ${payload.length}\r\n${extra}\r\n`),
		payload,
	]);
}

function assertBytesEqual(actual, expected) {
	// Keep failure diagnostics bounded: rendering a megabyte Buffer diff can
	// exhaust memory precisely when an owned subprocess fails or truncates output.
	assert.equal(actual.length, expected.length, "Byte lengths differ");
	assert.equal(actual.equals(expected), true, "Buffer contents differ");
}

// The maintained whole launcher runs in a VM. Only external process/filesystem
// boundaries are inert; its parsers, event handlers and lifecycle code are real.
// This same adapter runs the original source for genuine before/after witnesses.
function harness(realChild, requestedArgs = [], asMain = true) {
	const child = realChild || new EventEmitter();
	if (!realChild) {
		child.stdin = new PassThrough();
		child.stdout = new PassThrough();
		child.stderr = new PassThrough();
		child.kill = () => {
			child.killed = true;
			return true;
		};
	}
	let childClosed = false;
	child.once("close", () => {
		childClosed = true;
	});
	const received = [],
		output = [],
		errors = [],
		completions = [],
		spawnCalls = [];
	child.stdin.on("data", (chunk) => received.push(Buffer.from(chunk)));
	function collector(target) {
		return new Writable({
			write(chunk, _encoding, callback) {
				target.push(Buffer.from(chunk));
				callback();
			},
		});
	}
	const input = new PassThrough();
	const fakeProcess = Object.assign(new EventEmitter(), {
		argv: ["node", sourcePath, ...requestedArgs],
		env: {},
		stdin: input,
		stdout: collector(output),
		stderr: collector(errors),
		cwd: () => path.dirname(sourcePath),
		pid: 424242,
		exit: (code) => completions.push({ code, childClosed }),
		kill: () => completions.push({ code: 1, childClosed }),
	});
	Object.defineProperty(fakeProcess, "exitCode", {
		set(code) {
			completions.push({ code, childClosed });
		},
		get() {
			return completions.at(-1)?.code;
		},
	});
	const moduleRecord = { exports: {} };
	const externalRequire = (name) => {
		if (["child_process", "node:child_process"].includes(name)) {
			return {
				spawn: (binary, args, options) => {
					spawnCalls.push({ binary, args, options });
					return child;
				},
			};
		}
		if (["fs", "node:fs"].includes(name)) return { existsSync: () => true };
		return require(name);
	};
	externalRequire.main = asMain ? moduleRecord : {};
	vm.runInNewContext(
		source,
		{
			require: externalRequire,
			module: moduleRecord,
			exports: moduleRecord.exports,
			process: fakeProcess,
			Buffer,
			setTimeout,
			clearTimeout,
			setImmediate,
			console: {
				error: (...args) => fakeProcess.stderr.write(`${args.join(" ")}\n`),
			},
		},
		{ filename: sourcePath },
	);
	return {
		child,
		input,
		received,
		output,
		errors,
		completions,
		spawnCalls,
		outputStream: fakeProcess.stdout,
		receivedBytes: () => Buffer.concat(received),
		outputBytes: () => Buffer.concat(output),
		async finish(code = 0) {
			child.emit("exit", code, null);
			child.stdout.end();
			child.stderr.end();
			await tick();
			child.emit("close", code, null);
			await tick();
		},
	};
}

test("import is inert and ordinary launch arguments retain project semantics", async () => {
	const imported = harness(null, [], false);
	assert.equal(imported.spawnCalls.length, 0);
	for (const args of [
		[],
		["--project", "literal project"],
		["-P", "literal project"],
		["--project=literal project"],
	]) {
		const h = harness(null, args);
		assert.equal(h.spawnCalls.length, 1);
		const call = h.spawnCalls[0];
		assert.equal(
			call.binary,
			"T:\\projects\\serena-source\\target\\release\\serena.exe",
		);
		assert.deepEqual(
			Array.from(call.args),
			args.length ? args : ["--project", path.dirname(sourcePath)],
		);
		assert.equal(call.options.cwd, path.dirname(sourcePath));
		assert.equal(call.options.windowsHide, true);
		assert.deepEqual(Array.from(call.options.stdio), ["pipe", "pipe", "pipe"]);
		h.input.end();
		await h.finish();
	}
});

test("split framed header never becomes a JSON-line request", async () => {
	const h = harness();
	const payload = Buffer.from('{"id":9007199254740993,"text":"é🎧"}');
	const header = Buffer.from(`Content-Length: ${payload.length}\r\n`);
	h.input.write(header.subarray(0, 7));
	h.input.write(header.subarray(7));
	assert.equal(h.receivedBytes().length, 0);
	h.input.write(Buffer.from("\r\n"));
	h.input.write(payload.subarray(0, payload.length - 2));
	assert.equal(h.receivedBytes().length, 0);
	h.input.end(payload.subarray(payload.length - 2));
	assert.deepEqual(h.receivedBytes(), frame(payload));
	await h.finish();
	assert.equal(h.completions.at(-1).code, 0);
});

test("coalesced frames and byte-split Unicode preserve exact payloads", async () => {
	const h = harness();
	const one = Buffer.from('{"x":"🎧","n":9007199254740993}');
	const two = Buffer.from('[{"x":2}]');
	const bytes = Buffer.concat([
		frame(one, "Content-Type: application/json\r\n"),
		frame(two),
	]);
	for (const value of bytes) h.input.write(Buffer.from([value]));
	h.input.end();
	assert.deepEqual(h.receivedBytes(), bytes);
	h.child.stdout.end(Buffer.concat([frame(one), frame(two)]));
	await tick();
	await h.finish();
	assert.deepEqual(
		h.outputBytes(),
		Buffer.concat([one, Buffer.from("\n"), two, Buffer.from("\n")]),
	);
});

test("plain JSON input waits for newline or valid EOF without reserialization", async () => {
	const h = harness();
	const one = Buffer.from('  {"id":9007199254740993,"x":"é"}  ');
	h.input.write(one.subarray(0, 9));
	h.input.write(one.subarray(9));
	assert.equal(h.receivedBytes().length, 0);
	h.input.write(Buffer.from("\r\n"));
	assert.deepEqual(h.receivedBytes(), frame(one));
	const tail = Buffer.from('{"tail":true}');
	h.input.end(tail);
	await tick();
	assert.deepEqual(h.receivedBytes(), Buffer.concat([frame(one), frame(tail)]));
	await h.finish();
});

test("plain stdout JSON is never flushed at a chunk boundary", async () => {
	const h = harness();
	const payload = Buffer.from('{"id":9007199254740993,"x":"🎧"}');
	h.child.stdout.write(payload.subarray(0, 12));
	assert.equal(h.outputBytes().length, 0);
	h.child.stdout.write(payload.subarray(12));
	assert.equal(h.outputBytes().length, 0);
	h.child.stdout.end();
	await tick();
	await h.finish();
	assert.deepEqual(
		h.outputBytes(),
		Buffer.concat([payload, Buffer.from("\n")]),
	);
});

test("plain stdout coalescing preserves lines and trailing EOF JSON", async () => {
	const h = harness();
	h.child.stdout.write(Buffer.from('{"a":1}\n{"b":'));
	assert.deepEqual(h.outputBytes(), Buffer.from('{"a":1}\n'));
	h.child.stdout.end(Buffer.from("2}"));
	await tick();
	await h.finish();
	assert.deepEqual(h.outputBytes(), Buffer.from('{"a":1}\n{"b":2}\n'));
});

test("framed pretty JSON becomes one valid line retaining numeric and string lexemes", async () => {
	const h = harness();
	const payload = Buffer.from('{\n  "id": 9007199254740993, "x": "é🎧"\n}');
	const compact = Buffer.from('{"id":9007199254740993,"x":"é🎧"}\n');
	h.input.end(frame(payload));
	h.child.stdout.end(frame(payload));
	await tick();
	await h.finish();
	assert.deepEqual(h.receivedBytes(), frame(payload));
	assert.deepEqual(h.outputBytes(), compact);
	assert.deepEqual(JSON.parse(h.outputBytes()), JSON.parse(payload));
	assert.equal(h.outputBytes().filter((value) => value === 0x0a).length, 1);
});

test("output compaction preserves escaped quotes, backslashes, whitespace inside strings and UTF8", async () => {
	const h = harness();
	const strings = [
		String.raw`" é🎧 \"quoted\" \\ path \t \n "`,
		String.raw`"ends-with-backslash\\"`,
		String.raw`"slash-then-quote\\\"tail"`,
	];
	const payload = Buffer.from(
		` \r\n { \t "id" : 9007199254740993, "n" : 1.2300e+04, "s" : [\r\n ${strings.join(" , \n ")} \n ] } \r\n `,
	);
	const expected = Buffer.from(
		`{"id":9007199254740993,"n":1.2300e+04,"s":[${strings.join(",")}]}\n`,
	);
	h.child.stdout.end(frame(payload));
	await tick();
	await h.finish();
	assert.deepEqual(h.outputBytes(), expected);
	assert.deepEqual(JSON.parse(h.outputBytes()), JSON.parse(payload));
	assert.equal(h.outputBytes().filter((value) => value === 0x0a).length, 1);
	assert.equal(h.completions.at(-1).code, 0);
});

test("backpressure pauses input and resumes only after the child drain", async () => {
	const h = harness();
	h.child.stdin.removeAllListeners("data");
	h.child.stdin.pause();
	const payload = Buffer.from(JSON.stringify({ x: "a".repeat(200000) }));
	h.input.write(frame(payload));
	assert.equal(h.input.isPaused(), true);
	h.child.stdin.on("data", (chunk) => h.received.push(Buffer.from(chunk)));
	h.child.stdin.resume();
	await tick();
	await tick();
	assert.equal(h.input.isPaused(), false);
	assertBytesEqual(h.receivedBytes(), frame(payload));
	h.input.end();
	await h.finish();
});

const badFrames = [
	[
		"duplicate length",
		Buffer.from("Content-Length: 2\r\ncontent-length: 2\r\n\r\n{}"),
	],
	["missing length", Buffer.from("Content-Type: application/json\r\n\r\n{}")],
	["negative length", Buffer.from("Content-Length: -2\r\n\r\n{}")],
	["nonnumeric length", Buffer.from("Content-Length: 2oops\r\n\r\n{}")],
	["oversized length", Buffer.from("Content-Length: 8388609\r\n\r\n")],
	[
		"unsafe integer length",
		Buffer.from("Content-Length: 9007199254740993\r\n\r\n"),
	],
	["truncated header", Buffer.from("Content-Length: 2\r\n")],
	["truncated body", Buffer.from("Content-Length: 3\r\n\r\n{}")],
	["invalid JSON", frame(Buffer.from('{"broken":'))],
	["invalid UTF8", frame(Buffer.from([0x22, 0xff, 0x22]))],
	["oversized header", Buffer.from(`Content-Type: ${"a".repeat(16385)}`)],
];
for (const [name, bytes] of badFrames) {
	for (const direction of ["input", "output"]) {
		test(`${direction} refuses ${name} without publishing its payload`, async () => {
			const h = harness();
			if (direction === "input") h.input.end(bytes);
			else h.child.stdout.end(bytes);
			await tick();
			await h.finish();
			assert.notEqual(h.completions.at(-1).code, 0);
			assert.equal(
				direction === "input"
					? h.receivedBytes().length
					: h.outputBytes().length,
				0,
			);
			assert.ok(Buffer.concat(h.errors).length > 0);
		});
	}
}

for (const direction of ["input", "output"]) {
	test(`${direction} refuses truncated plain JSON at EOF`, async () => {
		const h = harness();
		if (direction === "input") h.input.end(Buffer.from('{"x":'));
		else h.child.stdout.end(Buffer.from('{"x":'));
		await tick();
		await h.finish();
		assert.notEqual(h.completions.at(-1).code, 0);
		assert.equal(
			direction === "input" ? h.receivedBytes().length : h.outputBytes().length,
			0,
		);
	});
}

test("JSON line buffering is bounded", async () => {
	const h = harness();
	h.input.end(Buffer.from(`"${"a".repeat(8388609)}`));
	await tick();
	await h.finish();
	assert.notEqual(h.completions.at(-1).code, 0);
	assert.equal(h.receivedBytes().length, 0);
});

test("declared message and header limits admit below/exact and reject above", async () => {
	for (const size of [8388607, 8388608, 8388609]) {
		const h = harness();
		const payload = Buffer.from(`"${"a".repeat(size - 2)}"`);
		h.input.end(payload);
		await tick();
		await h.finish();
		assert.equal(h.completions.at(-1).code, size <= 8388608 ? 0 : 1);
		assert.equal(
			h.receivedBytes().length,
			size <= 8388608 ? frame(payload).length : 0,
		);
	}
	const prefix = Buffer.from("Content-Length: 2\r\nX-Pad: ");
	for (const size of [16383, 16384, 16385]) {
		const h = harness();
		const bytes = Buffer.concat([
			prefix,
			Buffer.from("x".repeat(size - prefix.length - 4)),
			Buffer.from("\r\n\r\n{}"),
		]);
		h.input.end(bytes);
		await tick();
		await h.finish();
		assert.equal(h.completions.at(-1).code, size <= 16384 ? 0 : 1);
		assert.equal(h.receivedBytes().length, size <= 16384 ? bytes.length : 0);
	}
});

test("destination queue overflow fails closed and terminates only the owned child", async () => {
	const h = harness();
	Object.defineProperty(h.outputStream, "writableLength", {
		get: () => 2 * 8388608 + 16384,
	});
	h.child.stdout.end(frame(Buffer.from("{}")));
	await tick();
	await h.finish();
	assert.equal(h.completions.at(-1).code, 1);
	assert.equal(h.child.killed, true);
	assert.equal(h.outputBytes().length, 0);
});

test("child startup/pipe failure remains nonzero and does not publish stdout", async () => {
	for (const pipe of [false, true]) {
		const h = harness();
		(pipe ? h.child.stdin : h.child).emit(
			"error",
			new Error("inert child failure"),
		);
		await h.finish();
		assert.equal(h.completions.at(-1).code, 1);
		assert.equal(h.child.killed, true);
		assert.equal(h.outputBytes().length, 0);
		assert.match(Buffer.concat(h.errors).toString(), /inert child failure/);
	}
});

test("malformed JSON diagnostics do not echo payload content", async () => {
	const h = harness();
	h.input.end(frame(Buffer.from('{"secretCanary":"not-for-stderr",broken}')));
	await tick();
	await h.finish();
	assert.equal(h.completions.at(-1).code, 1);
	assert.match(Buffer.concat(h.errors).toString(), /Malformed JSON message/);
	assert.doesNotMatch(
		Buffer.concat(h.errors).toString(),
		/secretCanary|not-for-stderr|broken/,
	);
});

test("child exit does not complete before the final stdout drain", async () => {
	const h = harness();
	const payload = Buffer.from('{"last":true}');
	h.child.emit("exit", 0, null);
	assert.equal(h.completions.length, 0);
	h.child.stdout.end(frame(payload));
	await tick();
	h.child.emit("close", 0, null);
	await tick();
	assert.deepEqual(
		h.outputBytes(),
		Buffer.concat([payload, Buffer.from("\n")]),
	);
	assert.deepEqual(h.completions, [{ code: 0, childClosed: true }]);
});

for (const [name, bytes] of [
	["partial header", Buffer.from("Content-Length: 10\r\n")],
	["partial body", Buffer.from("Content-Length: 10\r\n\r\n{")],
	["partial JSON tail", Buffer.from('{"pending":')],
	["valid JSON tail without client EOF", Buffer.from('{"pending":true}')],
]) {
	test(`child close refuses pending inbound ${name} while client input remains open`, async () => {
		const h = harness();
		h.input.write(bytes);
		assert.equal(h.input.writableEnded, false);
		assert.equal(h.receivedBytes().length, 0);
		await h.finish();
		assert.equal(h.input.writableEnded, false);
		assert.deepEqual(h.completions, [{ code: 1, childClosed: true }]);
		assert.equal(h.receivedBytes().length, 0);
		assert.match(
			Buffer.concat(h.errors).toString(),
			/Truncated|Malformed|pending input/,
		);
	});
}

test("child close accepts open client input when no request remains pending", async () => {
	for (const bytes of [
		Buffer.alloc(0),
		frame(Buffer.from('{"complete":true}')),
	]) {
		const h = harness();
		if (bytes.length) h.input.write(bytes);
		await h.finish();
		assert.equal(h.input.writableEnded, false);
		assert.deepEqual(h.receivedBytes(), bytes);
		assert.deepEqual(h.completions, [{ code: 0, childClosed: true }]);
		assert.equal(h.errors.length, 0);
	}
});

test("signal termination uses the documented numeric convention after stdout drain", async () => {
	for (const [signal, expected] of [
		["SIGTERM", 143],
		["SIGKILL", 137],
	]) {
		const h = harness();
		const payload = Buffer.from('{"last":true}');
		h.child.emit("exit", null, signal);
		assert.equal(h.completions.length, 0);
		h.child.stdout.end(frame(payload));
		h.child.stderr.end();
		await tick();
		h.child.emit("close", null, signal);
		await tick();
		assertBytesEqual(
			h.outputBytes(),
			Buffer.concat([payload, Buffer.from("\n")]),
		);
		assert.deepEqual(h.completions, [{ code: expected, childClosed: true }]);
	}
});

for (const writeFails of [false, true]) {
	test(`completion waits for destination write callback (${writeFails ? "failure" : "success"})`, async () => {
		const h = harness();
		const payload = Buffer.from('{"last":true}');
		let settleWrite;
		let received;
		// Only the external output sink is delayed; the maintained transport is real.
		h.outputStream._write = (chunk, _encoding, callback) => {
			received = Buffer.from(chunk);
			settleWrite = callback;
		};
		h.child.stdout.write(frame(payload));
		await h.finish();
		assert.deepEqual(received, Buffer.concat([payload, Buffer.from("\n")]));
		assert.equal(typeof settleWrite, "function");
		assert.equal(h.completions.length, 0);
		settleWrite(writeFails ? new Error("destination-write-failure") : null);
		await tick();
		assert.deepEqual(h.completions, [
			{ code: writeFails ? 1 : 0, childClosed: true },
		]);
		if (writeFails)
			assert.match(
				Buffer.concat(h.errors).toString(),
				/destination-write-failure/,
			);
	});
}

test("owned real Node child drains bytes and preserves nonzero exit", {
	timeout: 10000,
}, async (context) => {
	const payload = Buffer.from(JSON.stringify({ text: "é🎧".repeat(20000) }));
	const script =
		"const b=Buffer.from(JSON.stringify({text:'é🎧'.repeat(20000)}));" +
		"process.stdin.resume();process.stdin.on('end',()=>{" +
		"process.stdout.write('Content-Length: '+b.length+'\\r\\n\\r\\n');" +
		"process.stdout.write(b,()=>{process.exitCode=23;});});";
	const child = spawn(process.execPath, ["-e", script], {
		stdio: ["pipe", "pipe", "pipe"],
		windowsHide: true,
	});
	context.signal.addEventListener("abort", () => child.kill("SIGKILL"), {
		once: true,
	});
	try {
		const h = harness(child);
		const closed = once(child, "close");
		h.input.end();
		await closed;
		await tick();
		assertBytesEqual(
			h.outputBytes(),
			Buffer.concat([payload, Buffer.from("\n")]),
		);
		assert.equal(h.completions.at(-1).code, 23);
		assert.equal(h.completions.at(-1).childClosed, true);
	} finally {
		if (child.exitCode === null) child.kill("SIGKILL");
	}
});

test("actual wrapper drains backpressured stdout with client stdin still open", {
	timeout: 10000,
}, async (context) => {
	const payload = Buffer.from(JSON.stringify({ text: "é🎧".repeat(200000) }));
	const childScript =
		"const b=Buffer.from(JSON.stringify({text:'é🎧'.repeat(200000)}));" +
		"process.stdout.write('Content-Length: '+b.length+'\\r\\n\\r\\n');" +
		"process.stdout.write(b,()=>{process.exitCode=23;});";
	const wrapperScript = `
		const fs=require('node:fs'), vm=require('node:vm'), cp=require('node:child_process');
		const script=fs.readFileSync(process.argv[1],'utf8');
		const record={exports:{}};
		function external(name){
			if(name==='child_process'||name==='node:child_process') return {spawn:()=>cp.spawn(process.execPath,['-e',${JSON.stringify(childScript)}],{stdio:['pipe','pipe','pipe'],windowsHide:true})};
			if(name==='fs'||name==='node:fs') return {existsSync:()=>true};
			return require(name);
		}
		external.main=record;
		vm.runInNewContext(script,{require:external,module:record,exports:record.exports,process,Buffer,console,setTimeout,clearTimeout,setImmediate});
	`;
	const wrapper = spawn(process.execPath, ["-e", wrapperScript, sourcePath], {
		stdio: ["pipe", "pipe", "pipe"],
		windowsHide: true,
	});
	context.signal.addEventListener("abort", () => wrapper.kill("SIGKILL"), {
		once: true,
	});
	const errors = [];
	wrapper.stderr.on("data", (chunk) => errors.push(chunk));
	try {
		const closed = once(wrapper, "close");
		wrapper.stdout.pause();
		await Promise.race([
			once(wrapper.stdout, "readable"),
			closed.then(([code, signal]) => {
				throw new Error(
					`Wrapper closed before output: code=${code} signal=${signal}; ${Buffer.concat(errors).toString()}`,
				);
			}),
		]);
		await new Promise((resolve) => setTimeout(resolve, 100));
		const chunks = [];
		wrapper.stdout.on("data", (chunk) => chunks.push(chunk));
		wrapper.stdout.resume();
		const [code] = await closed;
		assert.equal(code, 23, Buffer.concat(errors).toString());
		assertBytesEqual(
			Buffer.concat(chunks),
			Buffer.concat([payload, Buffer.from("\n")]),
		);
	} finally {
		wrapper.stdin.destroy();
		if (wrapper.exitCode === null) wrapper.kill("SIGKILL");
	}
});
