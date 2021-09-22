#include "common.h"
#include "project.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"
#include "antialias.h"
#include "optimize.h"
#include <ctime>

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024

Attribute pos;
Attribute texel;
Attribute normal;

RenderingParams p0;
ProjectParams pp0;
RasterizeParams rp0;
InterpolateParams ip0;
TexturemapParams tp0;
GLuint buffer0;
float* render0;
TexturemapParams tp1;
GLuint buffer1;
float* render1;
RenderingParams p2;
ProjectParams pp2;
RasterizeParams rp2;
InterpolateParams ip2;
TexturemapParams tp2;
GLuint buffer2;
float* render2;
TexturemapParams tp3;
GLuint buffer3;
float* render3;

LossParams loss;
AdamParams adam_tex;

static void init(void) {
	Rendering::init(p0, 512, 512, 1);
	loadOBJ("../../sphere.obj", pos, texel, normal);
	Project::forwardInit(pp0, pos);
	Rasterize::forwardInit(rp0, p0, pp0, pos, 1);
	Interpolate::forwardInit(ip0, p0, rp0, texel);
	Texturemap::forwardInit(tp0, p0, rp0, ip0, 2048, 1536, 3, 8);
	Texturemap::loadBMP(tp0, "../../earth-texture.bmp");
	Texturemap::buildMipTexture(tp0);
	Texturemap::forwardInit(tp1, p0, rp0, ip0, 2048, 1536, 3, 8);
	Loss::init(loss, tp1.out, tp0.out, p0.width * p0.height * 3, p0.width, p0.height, 3);
	Texturemap::backwardInit(tp1, p0, loss.grad);
	Adam::init(adam_tex, tp1.miptex[0], tp1.gradTex, tp1.width * tp1.height * tp1.channel, tp1.width, tp1.height, tp1.channel, 0.9, 0.999, 1e-3, 1e-8);


	Rendering::init(p2, 512, 512, 1);
	Project::forwardInit(pp2, pos);
	Rasterize::forwardInit(rp2, p2, pp2, pos, 1);
	Interpolate::forwardInit(ip2, p2, rp2, texel);
	Texturemap::forwardInit(tp2, p2, rp2, ip2, 2048, 1536, 3, 8);
	Texturemap::forwardInit(tp3, p2, rp2, ip2, 2048, 1536, 3, 8);
	for (int i = 0; i < tp0.miplevel; i++) {
		tp2.miptex[i] = tp0.miptex[i];
		tp3.miptex[i] = tp1.miptex[i];
	}

	Project::setProjection(pp0, 45, 1.0, 0.1, 10.0);
	Project::setView(pp0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	Project::setProjection(pp2, 45, 1.0, 0.1, 10.0);
	Project::setView(pp2, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	cudaMallocHost(&render0, p0.width * p0.height * 3 * sizeof(float));
	cudaMallocHost(&render1, p0.width * p0.height * 3 * sizeof(float));
	cudaMallocHost(&render2, p2.width * p2.height * 3 * sizeof(float));
	cudaMallocHost(&render3, p2.width * p2.height * 3 * sizeof(float));

	glGenTextures(1, &buffer0);
	glBindTexture(GL_TEXTURE_2D, buffer0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT15, buffer0, 0);
	glGenTextures(1, &buffer1);
	glBindTexture(GL_TEXTURE_2D, buffer1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT14, buffer1, 0);
	glGenTextures(1, &buffer2);
	glBindTexture(GL_TEXTURE_2D, buffer2);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT13, buffer2, 0);
	glGenTextures(1, &buffer3);
	glBindTexture(GL_TEXTURE_2D, buffer3);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT12, buffer3, 0);
}

static void display(void) {
	Project::forward(pp0);
	Rasterize::forward(rp0, p0);
	Interpolate::forward(ip0, p0);
	Texturemap::forward(tp0, p0);
	cudaMemcpy(render0, tp0.out, p0.width * p0.height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	Texturemap::forward(tp1, p0);
	Loss::backward(loss);
	Texturemap::backward(tp1, p0);
	Adam::step(adam_tex);
	Texturemap::buildMipTexture(tp1);

	cudaMemcpy(render1, tp1.out, p0.width * p0.height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	Project::forward(pp2);
	Rasterize::forward(rp2, p2);
	Interpolate::forward(ip2, p2);
	Texturemap::forward(tp2, p2);
	cudaMemcpy(render2, tp2.out, p2.width * p2.height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	Texturemap::forward(tp3, p2);
	cudaMemcpy(render3, tp3.out, p2.width * p2.height * 3 * sizeof(float), cudaMemcpyDeviceToHost);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, buffer0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, p0.width, p0.height, 0, GL_BGR, GL_FLOAT, render0);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(0.0f, -1.0f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, buffer1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, p0.width, p0.height, 0, GL_BGR, GL_FLOAT, render1);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, -1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, buffer2);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, p2.width, p2.height, 0, GL_BGR, GL_FLOAT, render2);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(0.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, buffer3);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, p2.width, p2.height, 0, GL_BGR, GL_FLOAT, render3);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
	glEnd();
	glFlush();
}

struct timespec pre, cur;
double t;
static void update(void) {
	pre = cur;
	timespec_get(&cur, TIME_UTC);
	long diff = cur.tv_nsec - pre.tv_nsec;
	if (diff < 0)diff = 1000000000 + cur.tv_nsec - pre.tv_nsec;
	double dt = (double)diff* pow(0.1, 9);
	t += dt;
	float theta = (float)rand() / (float)RAND_MAX * 360.0;
	float x = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	float y = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	float z = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;

	Project::setRotation(pp0, theta, x, y, z);
	Project::setRotation(pp2, t * 20, 0.0, 1.0, 0.0);
	printf("\r%3.3f ms:%3.3f fps %d step", dt*1000, 1.0 / dt,adam_tex.it);

	glutPostRedisplay();
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_RGBA);
	glutCreateWindow(argv[0]);
	glewInit();
	init();
	glutDisplayFunc(display);
	glutIdleFunc(update);
	glutMainLoop();
	return 0;
}