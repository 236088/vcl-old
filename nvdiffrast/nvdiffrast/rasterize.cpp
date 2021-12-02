#include "rasterize.h"

#define STRING(x) #x

static void compileShader(GLuint* shader, GLenum type, const GLchar* src)
{
	GLint compiled;
	GLsizei bufSize;

	*shader = glCreateShader(type);
	glShaderSource(*shader, 1, &src, NULL);
	glCompileShader(*shader);

	glGetShaderiv(*shader, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_FALSE) {
		fprintf(stderr, "Shader compile error.\n");
		exit(1);
	}

	glGetShaderiv(*shader, GL_INFO_LOG_LENGTH, &bufSize);
	if (bufSize > 1) {
		GLchar* infoLog = (GLchar*)malloc(bufSize);
		if (infoLog != NULL) {
			GLsizei length;
			glGetShaderInfoLog(*shader, bufSize, &length, infoLog);
			fprintf(stderr, "InfoLog:\n%s\n\n", infoLog);
			free(infoLog);
		}
		else
			fprintf(stderr, "Could not allocate InfoLog buffer.\n");
	}
}

static void linkProgram(GLuint* program, GLuint vertexShader, GLuint geometryShader, GLuint fragmentShader)
{
	GLsizei bufSize;
	GLint linked;

	*program = glCreateProgram();
	glAttachShader(*program, vertexShader);
	glAttachShader(*program, geometryShader);
	glAttachShader(*program, fragmentShader);
	glLinkProgram(*program);

	glGetProgramiv(*program, GL_LINK_STATUS, &linked);
	if (linked == GL_FALSE) {
		fprintf(stderr, "Link error.\n");
		exit(1);
	}

	glGetProgramiv(*program, GL_INFO_LOG_LENGTH, &bufSize);
	if (bufSize > 1) {
		GLchar* infoLog = (GLchar*)malloc(bufSize);
		if (infoLog != NULL) {
			GLsizei length;
			glGetProgramInfoLog(*program, bufSize, &length, infoLog);
			fprintf(stderr, "InfoLog:\n%s\n\n", infoLog);
			free(infoLog);
		}
		else
			fprintf(stderr, "Could not allocate InfoLog buffer.\n");
	}
}

void Rasterize::init(RasterizeParams& rp, ProjectParams& pp, int width, int height, int depth, int enableDB)
{
	rp.kernel.width = width;
	rp.kernel.height = height;
	rp.kernel.depth = depth;
	rp.kernel.proj = pp.kernel.out;
	rp.kernel.enableDB = enableDB;
	rp.kernel.xs = 2.f / float(width);
	rp.kernel.ys = 2.f / float(height);
	rp.kernel.idx =pp.vao;
	rp.projNum =pp.kernel.vboNum;
	rp.idxNum =pp.vaoNum;
	CUDA_ERROR_CHECK(cudaMallocHost(&rp.gl_proj,pp.kernel.vboNum * 4 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&rp.gl_idx,pp.vaoNum * 3* sizeof(unsigned int)));
	CUDA_ERROR_CHECK(cudaMemcpy(rp.gl_proj, rp.kernel.proj,pp.kernel.vboNum * 4 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(rp.gl_idx, rp.kernel.idx,pp.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMallocHost(&rp.gl_out, width * height * 4 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&rp.gl_outDB, width * height * 4 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&rp.kernel.out, width * height * 4 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&rp.kernel.outDB, width * height * 4 * sizeof(float)));
	rp.block = getBlock(width, height);
	rp.grid = getGrid(rp.block, width, height, depth);

	GLuint vertexShader;
	GLuint geometryShader;
	GLuint fragmentShader;

	compileShader(&vertexShader, GL_VERTEX_SHADER,
		"#version 430\n"
		STRING(
		layout(location = 0) in vec4 in_pos;
		void main() {
			gl_Position = in_pos;
		})
	);

	// calculate d[u, v]/d[pix]
	//
	// [p] = [p2 + (p0 - p2) * u + (p1 - p2) * v]
	// w = w2 + (w0 - w2) * u + (w1 - w2) * v
	//
	// [pix] = p / w <=> p = pix * w
	//
	// [(w0 * pix - p0) - (w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2)] * [u, v] = [-(w2 * pix - p2)]
	// [u, v] = [(w0 * pix - p0) - (w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2)]^-1 *[-(w2 * pix - p2)]
	//
	// D = det([(w0 * pix - p0) - (w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2)])
	//   = cross(w0 * pix - p0, w1 * pix - p1) - cross(w0 * pix - p0, w2 * pix - p2) - cross(w2 * pix - p2, w1 * pix - p1)
	//   = cross(w0 * p1 - w1 * p0, pix) - cross(w0 * p2 - w2 * p0, pix) + cross(w1 * p2 - w2 * p1, pix) + cross(p0, p1) - cross(p0, p2) + cross(p1, p2)
	//
	// u = cross(-((w1 * pix - p1) - (w2 * pix - p2)), -(w2 * pix - p2)) / D
	//   = cross((w1 * pix - p1), (w2 * pix - p2)) / D
	//   = (cross(w1 * p2 - w2 * p1, pix) + cross(p1, p2)) / D
	// v = (-cross(w0 * p2 - w2 * p0, pix) - cross(p0, p2)) / D
	//
	// dD/dx = (w0 * p1 - w1 * p0).y - (w0 * p2 - w2 * p0).y + (w1 * p2 - w2 * p1).y
	// dD/dy = -(w0 * p1 - w1 * p0).x + (w0 * p2 - w2 * p0).x - (w1 * p2 - w2 * p1).x
	//
	// du/dx = ((w1 * p2 - w2 * p1).y - u * dD/dx) / D
	//       = (u * ((w0 * p2 - w2 * p0).y - (w0 * p1 - w1 * p0).y) + (1 - u) * (w1 * p2 - w2 * p1).y) / D
	// du/dy = (u * (-(w0 * p2 - w2 * p0).x + (w0 * p1 - w1 * p0).x) + (1 - u) * -(w1 * p2 - w2 * p1).x) / D
	// dv/dx = (v * (-(w1 * p2 - w2 * p1).y - (w0 * p1 - w1 * p0).y) + (1 - v) * -(w0 * p2 - w2 * p0).y) / D
	// dv/dy = (v * ((w1 * p2 - w2 * p1).x + (w0 * p1 - w1 * p0).x) + (1 - v) * (w0 * p2 - w2 * p0).x) / D
	//
	// w * D = cross(p0, p1) * w2 - cross(p0, p2) * w1 + cross(p1, p2) * w0
	//       = cross(p0 * w2 - p2 * w0, p1 * w2 - p2 * w1) / w2
	if (enableDB) {
		compileShader(&geometryShader, GL_GEOMETRY_SHADER,
			"#version 430\n"
			STRING(
			layout(triangles) in;
			layout(triangle_strip, max_vertices = 3) out;
			layout(location = 0) uniform vec2 vp_scale;
			out vec4 var_uvzw;
			out vec4 var_db;
			void main() {

			float w0 = gl_in[0].gl_Position.w;
			float w1 = gl_in[1].gl_Position.w;
			float w2 = gl_in[2].gl_Position.w;
			vec2 p0 = gl_in[0].gl_Position.xy;
			vec2 p1 = gl_in[1].gl_Position.xy;
			vec2 p2 = gl_in[2].gl_Position.xy;

			vec2 e0 = w0 * p2 - w2 * p0;
			vec2 e1 = w1 * p2 - w2 * p1;
			vec2 e = w0 * p1 - w1 * p0;
			float a = e0.x * e1.y - e0.y * e1.x;

			float eps = 1e-6f;
			float ca = (abs(a) >= eps) ? a : (a < 0.f) ? -eps : eps;

			vec2 ascl = vp_scale.yx * w2 / ca;
			e0 *= ascl;
			e1 *= ascl;
			e *= ascl;

			gl_PrimitiveID = gl_PrimitiveIDIn;
			gl_Position = gl_in[0].gl_Position; var_uvzw = vec4(1.f, 0.f, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w); var_db = vec4(e0.y - e.y, -e0.x + e.x, -e0.y, e0.x); EmitVertex();
			gl_Position = gl_in[1].gl_Position; var_uvzw = vec4(0.f, 1.f, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w); var_db = vec4(e1.y, -e1.x, -e1.y - e.y, e1.x + e.x); EmitVertex();
			gl_Position = gl_in[2].gl_Position; var_uvzw = vec4(0.f, 0.f, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w); var_db = vec4(e1.y, -e1.x, -e0.y, e0.x); EmitVertex();
			})
		);
		compileShader(&fragmentShader, GL_FRAGMENT_SHADER,
			"#version 430\n"
			STRING(
			in vec4 var_uvzw;
			in vec4 var_db;
			layout(location = 0) out vec4 out_raster;
			layout(location = 1) out vec4 out_db;
			void main() {
				out_raster = vec4(var_uvzw.x, var_uvzw.y, var_uvzw.z / var_uvzw.w, float(gl_PrimitiveID + 1));
				out_db = var_db * var_uvzw.w;
			})
		);
	}
	else {
		compileShader(&geometryShader, GL_GEOMETRY_SHADER,
			"#version 430\n"
			STRING(
			layout(triangles) in;
			layout(triangle_strip, max_vertices = 3) out;
			out vec4 var_uvzw;
			void main() {
				gl_PrimitiveID = gl_PrimitiveIDIn;
				gl_Position = gl_in[0].gl_Position; var_uvzw = vec4(1.f, 0.f, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w); EmitVertex();
				gl_Position = gl_in[1].gl_Position; var_uvzw = vec4(0.f, 1.f, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w); EmitVertex();
				gl_Position = gl_in[2].gl_Position; var_uvzw = vec4(0.f, 0.f, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w); EmitVertex();
			})
		);
		compileShader(&fragmentShader, GL_FRAGMENT_SHADER,
			"#version 430\n"
			STRING(
			in vec4 var_uvzw;
			layout(location = 0) out vec4 out_raster;
			void main() {
				out_raster = vec4(var_uvzw.x, var_uvzw.y, var_uvzw.z / var_uvzw.w, float(gl_PrimitiveID + 1));
			})
		);
	}
	linkProgram(&rp.program, vertexShader, geometryShader, fragmentShader);

	glGenFramebuffers(1, &rp.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, rp.fbo);

	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(enableDB ? 2 : 1, DrawBuffers);

	glGenTextures(1, &rp.buffer);
	glBindTexture(GL_TEXTURE_2D, rp.buffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rp.gl_out);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, rp.buffer, 0);

	if (enableDB) {
		glGenTextures(1, &rp.bufferDB);
		glBindTexture(GL_TEXTURE_2D, rp.bufferDB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rp.gl_outDB);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rp.bufferDB, 0);
	}

	GLuint depthbuffer;
	glGenRenderbuffers(1, &depthbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuffer);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

	GLuint tri;
	glGenBuffers(1, &tri);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tri);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepth(1.0);
}

void Rasterize::forward(RasterizeParams& rp) {
	CUDA_ERROR_CHECK(cudaMemcpy(rp.gl_proj, rp.kernel.proj, rp.projNum * 4 * sizeof(float), cudaMemcpyDeviceToHost));
	glBindFramebuffer(GL_FRAMEBUFFER, rp.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(rp.program);
	glViewport(0, 0, rp.kernel.width, rp.kernel.height);
	glBufferData(GL_ARRAY_BUFFER, rp.projNum * 4 * sizeof(float), rp.gl_proj, GL_DYNAMIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, rp.idxNum * 3 * sizeof(float), rp.gl_idx, GL_DYNAMIC_DRAW);
	if (rp.kernel.enableDB)glUniform2f(0, rp.kernel.xs, rp.kernel.ys);
	glDrawElements(GL_TRIANGLES, rp.idxNum * 3, GL_UNSIGNED_INT, 0);

	glBindTexture(GL_TEXTURE_2D, rp.buffer);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, rp.gl_out);
	CUDA_ERROR_CHECK(cudaMemcpy(rp.kernel.out, rp.gl_out, rp.kernel.width * rp.kernel.height * 4 * sizeof(float), cudaMemcpyHostToDevice));

	if (rp.kernel.enableDB) {
		glBindTexture(GL_TEXTURE_2D, rp.bufferDB);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, rp.gl_outDB);
		CUDA_ERROR_CHECK(cudaMemcpy(rp.kernel.outDB, rp.gl_outDB, rp.kernel.width * rp.kernel.height * 4 * sizeof(float), cudaMemcpyHostToDevice));
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glFlush();
}
