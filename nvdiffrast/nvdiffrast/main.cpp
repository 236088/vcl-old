#include "preset.h"
#include <ctime>

//PresetFilter preset;
//PresetMaterial preset;
//PresetEarth preset;
//PresetCube preset;
PresetPrimitives preset;

struct timespec pre, cur;
double t;
float loss_sum = 0;
int count = 0;
int interbal = 100;
bool play = false;

static void InitFunc()
{
	timespec_get(&cur, TIME_UTC);
	srand(cur.tv_nsec);
	preset.init();
}

static void DisplayFunc(void)
{
	preset.display();
	loss_sum += preset.getLoss();
	if ((++count) % interbal == 0) {
		printf(" count: %d, loss: %f\n", count, loss_sum / interbal);
		loss_sum = 0;
	}
	if (count % 10000 == 0)play = false;
}

static void IdleFunc(void) 
{	
	if (!play)return;
	preset.update();
	pre = cur;
	timespec_get(&cur, TIME_UTC);
	long diff = cur.tv_nsec - pre.tv_nsec;
	if (diff < 0)diff = 1000000000 + cur.tv_nsec - pre.tv_nsec;
	double dt = (double)diff * 1e-9;
	t += dt;

	printf("\r%3.3f ms:%3.3f fps", dt * 1000, 1.0 / dt);
	glutPostRedisplay();
}

static void KeyboardFunc(unsigned char key, int x, int y) {
	switch (key)
	{
	case ' ':
		play = !play;
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitWindowSize(preset.windowWidth, preset.windowHeight);
	glutInitDisplayMode(GLUT_RGBA);
	glutCreateWindow(argv[0]);
	glewInit();
	InitFunc();
	glutDisplayFunc(DisplayFunc);
	glutIdleFunc(IdleFunc);
	glutKeyboardFunc(KeyboardFunc);
	glutMainLoop();
}