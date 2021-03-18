//============================================================================
// Name        : FinalProject.cpp
// Author      : Michael Quilici
// Course      : CS-330-J6917 Comp Graphic and Visualization 20EW6
// Date        : 8/14/2020
// Description : Final Project
//============================================================================
//
// Controls: Press ALT and use left mouse button to rotate
// Controls: Press ALT and use right mouse button to move in and out
// Controls: Press p to turn on/off perspective
//
//============================================================================

#include "GL/glew.h"     // Includes glew header
#include <GL/freeglut.h> // Includes freeglut header
#include "SOIL2/SOIL2.h" // Includes SOIL2 texture loader (put in src folder)

#include <iostream>      // Includes C++ i/o stream
#include <vector>        // Includes vector header

// GLM Math Header Inclusions
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/string_cast.hpp"

using namespace std; // Use standard namespace

#define WINDOW_TITLE "OpenGL Demo" // Define window title

// Vertex and Fragment Shader source macro
#ifndef GLSL
#define GLSL(Version, Source) "#version " #Version "\n" #Source
#endif

// Object and texture file names
const char* ObjectFile   = "chair.obj";
const char* TextureFile1 = "wood.jpg";
const char* TextureFile2 = "wood_normal.jpg";
const char* TextureFile3 = "floor.jpg";
const char* TextureFile4 = "floor_normal.jpg";
const char* TextureFile5 = "sky.jpg";

// Global variables
GLuint VBO[4], VAO[4], texture[6]; // Arrays to hold buffer and texture objects
GLuint specularShader;             // Specular shader
GLuint diffuseShader;              // Diffuse shader
GLuint lampShader;                 // Lamp shader
GLuint depthShader;                // Depth map shader
GLuint debugDepthShader;           // Shadow map shader
GLuint depthMapFBO;                // Shadow map frame buffer
GLuint WINDOW_WIDTH = 1200;        // Define window width
GLuint WINDOW_HEIGHT = 800;        // Define window height
GLuint SHADOW_WIDTH = 1024;        // Define shadow map width
GLuint SHADOW_HEIGHT = 1024;       // Define shadow map height
GLfloat cameraRotV = 80;           // Camera vertical rotation angle
GLfloat cameraRotH = 200;          // Camera horizontal rotation angle
GLfloat cameraDistance = 4.0f;     // Camera distance from origin
GLfloat near_plane = 0.01f;        // Shadow extents
GLfloat far_plane = 15.0f;         // Shadow extents
GLint mousex, mousey;              // Mouse coordinates
GLboolean orbitCamera = GL_FALSE;  // Flag for rotating the object
GLboolean zoomCamera = GL_FALSE;   // Flag for rotating the camera
GLboolean keyModifier = GL_FALSE;  // Flag for pressing ALT button
GLboolean perspective = GL_TRUE;   // Flag for enabling perspective

// Camera Attributes
glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraUpY = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);

// Key Light Attributes
glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
glm::vec3 lightPosition = glm::vec3(-2.0f, 1.0f, 0.0f);

// Object vertices, UVs and normals loaded from file
std::vector<glm::vec3> objVertices;
std::vector<glm::vec2> objUVs;
std::vector<glm::vec3> objNormals;
std::vector<glm::vec3> objTangents;   // Tangent vectors used for normal mapping
std::vector<glm::vec3> objBitangents; // Bitangent vectors used for normal mapping
std::vector<int> faces;

// Function prototypes
void UResizeWindow(int, int);
void URenderGraphics(void);
void UCreateShader(void);
void UCreateBuffers(void);
void UGenerateTexture(void);
void UKeyPressed(unsigned char, int, int);
void USpecialKeyPressed(int, int, int);
void USpecialKeyReleased(int, int, int);
void UKeyReleased(unsigned char, int, int);
void UMousePress(int, int, int, int);
void UMouseMotion(int, int);
void ULoadOBJ(const char*);

/* Specular vertex shader source */
const GLchar * specularVertexShader = GLSL(330,

	layout (location=0) in vec3 position;           // Get vertex position from attribute 0
	layout (location=1) in vec3 normal;             // Get normal vector from attribute 1
	layout (location=2) in vec2 textureCoordinates; // Get texture coordinates from attribute 2
	layout (location=3) in vec3 tangent;            // Get tangent vector from attribute 3
	layout (location=4) in vec3 bitangent;          // Get bitangent vector from attribute 4

	out vec2 TexCoords;           // Outgoing texture coordinates to fragment shader
	out vec3 FragmentPos;         // Outgoing fragment position to fragment shader
    out vec3 Normal;              // Outgoing normal vector
	out vec3 TangentLightPos;     // Outgoing tangent space key light position to fragment shader
	out vec3 TangentViewPos;      // Outgoing tangent space view position to fragment shader
	out vec3 TangentFragPos;      // Outgoing tangent space fragment position to fragment shader
	out vec4 FragPosLightSpace;   // Outgoing clip space light position

	uniform mat4 model;           // 4x4 model matrix to transform vertices from model space into world space
	uniform mat4 view;            // 4x4 view matrix to transform vertices from world space to view space
	uniform mat4 projection;      // 4x4 projection matrix to transform vertices from view space to clip space
	uniform vec3 viewPosition;    // Uniform variable holds view position
	uniform vec3 lightPos;        // Uniform variable holds key light position
	uniform mat4 lightView;       // Uniform variable holds light view matrix
	uniform mat4 lightProjection; // Uniform variable holds light projection matrix

	void main() {
		// Flip texture vertical
		TexCoords = vec2(textureCoordinates.x, 1.0f - textureCoordinates.y);

		// Get fragment position in world space
		FragmentPos = vec3(model * vec4(position, 1.0f));

		// Compute normal matrix
		mat3 normalMatrix = transpose(inverse(mat3(model)));

		// Get normal vector in world space and exclude translation properties
		vec3 N = normalize(normalMatrix * normal);

		// Get tangent vector in world space and exclude translation properties
		vec3 T = normalize(normalMatrix * tangent);

		// Get bitangent vector in world space and exclude translation properties
		vec3 B = normalize(normalMatrix * bitangent);
		//vec3 B = cross(N, T);

		// Compute TBN matrix used to transform vectors into tangent space
		mat3 TBN = transpose(mat3(T, B, N));

		// Compute lighting position, view position and fragment position in tangent space
		TangentLightPos  = TBN * lightPos;
		TangentViewPos   = TBN * viewPosition;
		TangentFragPos   = TBN * FragmentPos;

	    Normal = normal;
	    //Normal = TBN * N;

	    // Clip space position of fragment from light's perspective
	    FragPosLightSpace = lightProjection * lightView * vec4(FragmentPos, 1.0f);

		gl_Position = projection * view * model * vec4(position, 1.0f); // Transform vertex data using matrix

	}
);

/* Specular fragment Shader source */
const GLchar * specularFragmentShader = GLSL(330,

	in vec2 TexCoords;           // Get texture coordinates from vertex shader
	in vec3 FragmentPos;         // Get fragment position from vertex shader
	in vec3 Normal;              // Get normal vector from vertex shader
	in vec3 TangentLightPos;     // Get tangent space key light position
	in vec3 TangentLightPos2;    // Get tangent space fill light position
	in vec3 TangentViewPos;      // Get tangent space view position
	in vec3 TangentFragPos;      // Get tangent space fragment position
	in vec4 FragPosLightSpace;   // Get fragment position for light

	out vec4 objColor;           // Output fragment color

	uniform sampler2D uTexture;  // Uniform sampler for texture map
	uniform sampler2D uNormal;   // Uniform sampler for normal map
	uniform sampler2D shadowMap; // Uniform sampler for shadow map
	uniform vec3 viewPosition;   // Uniform variable for view position
	uniform vec3 lightColor;     // Uniform variable for key light color
	uniform vec3 lightPos;       // Uniform variable for key light position

	// https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
	float ShadowCalculation(vec4 fragPosLightSpace)
	{
	    // perform perspective divide
	    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	    // transform to [0,1] range
	    projCoords = projCoords * 0.5 + 0.5;
	    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
	    float closestDepth = texture(shadowMap, projCoords.xy).r;
	    // get depth of current fragment from light's perspective
	    float currentDepth = projCoords.z;
	    // calculate bias (based on depth map resolution and slope)
	    vec3 normal = normalize(Normal);
	    vec3 lightDir = normalize(lightPos - FragmentPos);
	    float bias = 0.0f; //max(0.005 * (1.0 - dot(Normal, lightDir)), 0.001);
	    // check whether current frag pos is in shadow
	    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
	    // PCF
	    float shadow = 0.0;
	    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
	    for(int x = -1; x <= 1; ++x)
	    {
	        for(int y = -1; y <= 1; ++y)
	        {
	            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
	            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;
	        }
	    }
	    shadow /= 9.0;

	    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
	    if(projCoords.z > 1.0)
	        shadow = 0.0;

	    return shadow;
	}


	void main(){

		// Key light properties
		float ambientIntensity   = 0.2f;  // Ambient intensity
		float diffuseIntensity  = 1.0f;   // Diffuse intensity
		float specularIntensity = 1.5f;   // Specular intensity
		float highlightSize     = 128.0f; // Shininess

	    // Get diffuse color
	    vec4 color = texture(uTexture, TexCoords);

	    // Obtain normal from normal map in range [0,1]
	    vec3 normal = texture(uNormal, TexCoords).rgb;

	    // Transform normal vector to range [-1,1]
	    normal = normalize(normal * 2.0 - 1.0);

	    // Ambient
	    vec3 ambient = ambientIntensity * lightColor;

	    // Diffuse
	    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
	    float diff = max(dot(lightDir, normal), 0.0);
	    vec3 diffuse = diff * diffuseIntensity * lightColor;

	    // Specular
	    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
	    vec3 reflectDir = reflect(-lightDir, normal);
	    vec3 halfwayDir = normalize(lightDir + viewDir);
	    float spec = pow(max(dot(normal, halfwayDir), 0.0), highlightSize);
	    vec3 specular = vec3(specularIntensity) * spec * lightColor;

	    // Compute inverse-square law light attenuation
	    float distance    = length(lightPos - FragmentPos);
	    float attenuation = 15.0 / (distance * distance);

	    // Calculate shadow
	    float shadow = ShadowCalculation(FragPosLightSpace);

	    vec4 phong = vec4(ambient + (1.0 - shadow) * (diffuse + specular), 1.0);

	    objColor = attenuation * phong * color;
	}
);

/* Diffuse vertex shader source */
const GLchar * diffuseVertexShader = GLSL(330,
	layout (location=0) in vec3 position; // Get vertex coordinates from attribute 0
    layout (location=1) in vec2 textureCoordinates;

	out vec2 mobileTextureCoordinate; // Outgoing texture coordinates to fragment shader

	uniform mat4 model;      // 4x4 model matrix to transform vertices from model space into world space
	uniform mat4 view;       // 4x4 view matrix to transform vertices from world space to view space
	uniform mat4 projection; // 4x4 projection matrix to transform vertices from view space to clip space

	void main() {
		mobileTextureCoordinate = vec2(textureCoordinates.x, 1.0f - textureCoordinates.y); // Flip texture vertical
		gl_Position = projection * view * model * vec4(position, 1.0f); // Transform vertex data using matrices
	}
);

/* Diffuse fragment shader source */
const GLchar * diffuseFragmentShader = GLSL(330,

	in vec2 mobileTextureCoordinate; // Get texture coordinates from vertex shader

	out vec4 gpuTexture; // Variable to transfer color data to the GPU

	uniform sampler2D uTexture; // Uniform sampler for texture map

	void main(){
		gpuTexture = texture(uTexture, mobileTextureCoordinate);
	}
);

/* Lamp vertex shader source */
const GLchar * lampVertexShader = GLSL(330,

	layout (location=0) in vec3 position; // Get vertex coordinates from attribute 0

	uniform mat4 model;      // 4x4 model matrix to transform vertices from model space into world space
	uniform mat4 view;       // 4x4 view matrix to transform vertices from world space to view space
	uniform mat4 projection; // 4x4 projection matrix to transform vertices from view space to clip space

	void main() {
		gl_Position = projection * view * model * vec4(position, 1.0f); // Transform vertex data using matrices
	}
);

/* Lamp fragment shader source */
const GLchar * lampFragmentShader = GLSL(330,

	out vec4 color; // Variable to transfer color data to the GPU

	uniform vec3 lightColor; // Uniform variable holds lamp color

	void main(){
		color = vec4(lightColor, 1.0f); // Send color data to GPU
	}
);

/* Shadow vertex shader source */
const GLchar * shadowVertexShader = GLSL(330,
	layout (location=0) in vec3 position; // Get vertex coordinates from attribute 0

	uniform mat4 model;      // 4x4 model matrix to transform vertices from model space into world space
	uniform mat4 view;       // 4x4 view matrix to transform vertices from world space to view space
	uniform mat4 projection; // 4x4 projection matrix to transform vertices from view space to clip space

	void main() {
		gl_Position = projection * view * model * vec4(position, 1.0f); // Transform vertex data using matrices
	}
);

/* Shadow fragment shader source */
const GLchar * shadowFragmentShader = GLSL(330,
	void main(){
	}
);

/* Debug depth ertex shader source */
const GLchar * debugVertexShader = GLSL(330,
	layout (location = 0) in vec3 position;           // Get vertex position from attribute 0
	layout (location = 1) in vec2 textureCoordinates; // Get texture coordinates from attribute 1

	out vec2 mobileTextureCoordinate; // Outgoing texture coordinates to fragment shader

	void main()
	{
		mobileTextureCoordinate = textureCoordinates; // Send texture coordinates to fragment shader
		gl_Position = vec4(position, 1.0);            // Transform vertex data using matrices
	}
);

/* Debug depth fragment shader source */
const GLchar * debugFragmentShader = GLSL(330,
	out vec4 FragColor;

	in vec2 mobileTextureCoordinate;

	uniform sampler2D depthMap; // Uniform sampler for depth map
	uniform float near_plane;   // Uniform for near clipping distance
	uniform float far_plane;    // Uniform for far clipping distance

	// Required when using a perspective projection matrix
	float LinearizeDepth(float depth)
	{
		float z = depth * 2.0 - 1.0; // Back to NDC
		return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
	}

	void main()
	{
		float depthValue = texture(depthMap, mobileTextureCoordinate).r;     // Get depth value from texture
		FragColor = vec4(vec3(LinearizeDepth(depthValue) / far_plane), 1.0); // Perspective projection for point light
		//FragColor = vec4(vec3(depthValue), 1.0);                           // Orthographic projection for distance light
	}
);

/* Main method */
int main(int argc, char *argv[]) {

	glutInit(&argc, argv); // Initializes freeglut library

	// Enable window depth buffer, double buffering, RGBA mode, and multisampling
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);

	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT); // Specify window dimensions

	// Center window using screen width and current width
	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - WINDOW_WIDTH) / 2,
			(glutGet(GLUT_SCREEN_HEIGHT) - WINDOW_HEIGHT) / 2);

	glutCreateWindow(WINDOW_TITLE); // Create a window and title

	glutReshapeFunc(UResizeWindow); // Set display callback for window resize event

	// Check for error initializing GLEW
	glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK) {
			std::cout << "Failed to initialize GLEW" << std::endl;
			return -1;
		}

	UCreateShader();    // Create shaders
	UGenerateTexture(); // Create texture
	UCreateBuffers();   // Create buffers

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);   // Set the background color

	glutDisplayFunc(URenderGraphics);       // Set display callback for window resize event

	// User Input
	glutKeyboardFunc(UKeyPressed);          // Set callback for key press
	glutKeyboardUpFunc(UKeyReleased);       // Set callback for key release
	glutSpecialFunc(USpecialKeyPressed);    // Set callback for special key press (ALT)
	glutSpecialUpFunc(USpecialKeyReleased); // Set callback for special key release (ALT)
	glutMouseFunc(UMousePress);             // Set callback for mouse button press
	glutMotionFunc(UMouseMotion);           // Set callback for mouse movement

	glutMainLoop(); // Enter event processing loop

	// Destroy buffer objects once used
	for (int i=0; i<4; ++i) {
		glDeleteVertexArrays(1, &VAO[i]);
		glDeleteBuffers(1, &VBO[i]);
	}

	return 0;
}

/* Callback handler for resizing the window */
void UResizeWindow(int w, int h) {
	WINDOW_WIDTH = w;
	WINDOW_HEIGHT = h;
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT); // Set the viewport size
}

/* Callback handler for rendering graphics */
void URenderGraphics(void) {

	glEnable(GL_DEPTH_TEST); // Enables depth testing for correct 3D rendering

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the screen

    glEnable(GL_CULL_FACE);

//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Rotation angles to rotate object
	GLfloat angleh = cameraRotH*M_PI/180;
	GLfloat anglev = cameraRotV*M_PI/180;

	// Transform the camera
	cameraPosition[0] = cameraDistance * sin(anglev) * cos(angleh);
	cameraPosition[1] = cameraDistance * cos(anglev);
	cameraPosition[2] = cameraDistance * sin(anglev) * sin(angleh);

	/** Animate the rotation of the object **/
	GLfloat time = glutGet(GLUT_ELAPSED_TIME);
	float t = 0.0005f*time;

	// Transform the light
	lightPosition[0] = 3.0f * cos(t);
	lightPosition[1] = 2.0f;
	lightPosition[2] = 3.0f * sin(t);

	// 1. ----- Render Depth Map -----

	glCullFace(GL_FRONT);

    // reset viewport
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(depthShader);
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);

	glClear(GL_DEPTH_BUFFER_BIT);

	glm::mat4 lightModel(1.0f);

	glm::mat4 lightView = glm::lookAt(lightPosition, cameraTarget, cameraUpY);

	GLfloat lightAspect = (GLfloat)SHADOW_WIDTH / (GLfloat)SHADOW_HEIGHT;

	//glm::mat4 lightProjection = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, near_plane, far_plane);
	glm::mat4 lightProjection = glm::perspective(glm::radians(45.0f), lightAspect, near_plane, far_plane); // note that if you use a perspective projection matrix you'll have to change the light position as the current light position isn't enough to reflect the whole scene

	// Get the integers that represent location of each uniform in the vertex shader
	GLint lightModelLoc = glGetUniformLocation(depthShader, "model");
	GLint lightViewLoc  = glGetUniformLocation(depthShader, "view");
	GLint lightProjLoc  = glGetUniformLocation(depthShader, "projection");

	// Send transform information to the vertex shader
	glUniformMatrix4fv(lightModelLoc, 1, GL_FALSE, glm::value_ptr(lightModel));
	glUniformMatrix4fv(lightViewLoc, 1, GL_FALSE, glm::value_ptr(lightView));
	glUniformMatrix4fv(lightProjLoc, 1, GL_FALSE, glm::value_ptr(lightProjection));

	// Chair
	glBindVertexArray(VAO[0]);
	glDrawArrays(GL_TRIANGLES, 0, objVertices.size());
	glBindVertexArray(0);

	// Floor
	glBindVertexArray(VAO[1]);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// 2. ----- Render Scene -----

	glCullFace(GL_BACK); // don't forget to reset original culling face

    // reset viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(specularShader);

	glm::mat4 model(1.0f);
	glm::mat4 view = glm::lookAt(cameraPosition, cameraTarget, cameraUpY);

	GLfloat aspect = (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT;

	glm::mat4 projection(1.0f);
	if (perspective) {
		projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 9999.0f);
	} else {
		projection = glm::ortho(-cameraDistance/2*aspect, cameraDistance/2*aspect, -cameraDistance/2, cameraDistance/2, 0.1f, 9999.0f);
	}

	// Get the integers that represent location of each uniform in the vertex shader
	GLint modelLoc = glGetUniformLocation(specularShader, "model");
	GLint viewLoc = glGetUniformLocation(specularShader, "view");
	GLint projLoc = glGetUniformLocation(specularShader, "projection");

	// Send transform information to the vertex shader
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

	// Set camera position
	GLint viewPositionLoc = glGetUniformLocation(specularShader, "viewPosition");
	glUniform3f(viewPositionLoc, cameraPosition.x, cameraPosition.y, cameraPosition.z);

	// Set light properties
	lightViewLoc = glGetUniformLocation(specularShader, "lightView");
	lightProjLoc = glGetUniformLocation(specularShader, "lightProjection");
	glUniformMatrix4fv(lightViewLoc, 1, GL_FALSE, glm::value_ptr(lightView));
	glUniformMatrix4fv(lightProjLoc, 1, GL_FALSE, glm::value_ptr(lightProjection));

	GLint lightColorLoc = glGetUniformLocation(specularShader, "lightColor");
	GLint lightPositionLoc = glGetUniformLocation(specularShader, "lightPos");
	glUniform3f(lightColorLoc, lightColor.r, lightColor.g, lightColor.b);
	glUniform3f(lightPositionLoc, lightPosition.x, lightPosition.y, lightPosition.z);

    // Draw chair
	glUniform1i(glGetUniformLocation(specularShader, "uTexture" ), 0);  // Load texture uniform
	glUniform1i(glGetUniformLocation(specularShader, "uNormal"  ), 1);   // Load texture uniform
	glUniform1i(glGetUniformLocation(specularShader, "shadowMap"), 5); // Load texture uniform
	glBindVertexArray(VAO[0]); // Activate the vertex array object
	glDrawArrays(GL_TRIANGLES, 0, objVertices.size()); // Draw the triangles
	glBindVertexArray(0); // Deactivate the vertex array object

	// Draw floor
	model = glm::mat4(1.0f);
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
	glUniform1i(glGetUniformLocation(specularShader, "uTexture" ), 2);  // Load texture uniform
	glUniform1i(glGetUniformLocation(specularShader, "uNormal"  ), 3);   // Load texture uniform
	glUniform1i(glGetUniformLocation(specularShader, "shadowMap"), 5); // Load texture uniform
	glBindVertexArray(VAO[1]); // Activate the vertex array object
	glDrawArrays(GL_TRIANGLES, 0, 6); // Draw the triangles
	glBindVertexArray(0); // Deactivate the vertex array object

	// Draw lamp
	glUseProgram(lampShader); // Use lamp shader
	glm::vec3 lampScale(0.001f); // Set lamp scale
	model = glm::mat4(1.0f);                      // Create 4x4 model matrix
	model = glm::translate(model, lightPosition); // Translate object to light position
	model = glm::scale(model, lampScale);         // Scale object
	modelLoc = glGetUniformLocation(lampShader, "model");
	viewLoc = glGetUniformLocation(lampShader, "view");
	projLoc = glGetUniformLocation(lampShader, "projection");
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
	lightColorLoc = glGetUniformLocation(lampShader, "lightColor");
	glUniform3f(lightColorLoc, lightColor.r, lightColor.g, lightColor.b);
	glBindVertexArray(VAO[2]); // Activate key light lamp VAO
	glDrawArrays(GL_TRIANGLES, 0, 36); // Draw key light lamp
	glBindVertexArray(0); // Deactivate vertex array object

	// Draw sky
	glCullFace(GL_FRONT);
	glUseProgram(diffuseShader);
	model = glm::mat4(1.0f);
	modelLoc = glGetUniformLocation(diffuseShader, "model");
	viewLoc  = glGetUniformLocation(diffuseShader, "view");
	projLoc  = glGetUniformLocation(diffuseShader, "projection");
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(projection));
	viewPositionLoc = glGetUniformLocation(diffuseShader, "viewPosition");
	glUniform3f(viewPositionLoc, cameraPosition.x, cameraPosition.y, cameraPosition.z);
	glm::mat4 skyView = glm::mat4(glm::mat3(view)); // Remove translation from sky view
	glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(skyView));
	glUniform1i(glGetUniformLocation(diffuseShader, "uTexture"), 4); // Load texture uniform
	glBindVertexArray(VAO[2]); // Activate the vertex array object
	glDrawArrays(GL_TRIANGLES, 0, 36); // Draw the triangles
	glBindVertexArray(0); // Deactivate the vertex array object


	bool testDepthFrame = 0;

	// 3. ----- Render Depth Test Quad -----
	if (testDepthFrame) {
		glCullFace(GL_BACK); // don't forget to reset original culling face
		glUseProgram(debugDepthShader);
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, texture[5]);
		glUniform1i(glGetUniformLocation(debugDepthShader, "depthMap"), 5); // Load texture uniform
		GLint nearplaneloc = glGetUniformLocation(debugDepthShader, "near_plane");
		GLint farplaneloc = glGetUniformLocation(debugDepthShader, "far_plane");
		glUniform1f(nearplaneloc, near_plane);
		glUniform1f(farplaneloc, far_plane);
		glBindVertexArray(VAO[3]);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0); // Deactivate the vertex array object
	}

	glutPostRedisplay(); // Redisplay viewport
	glutSwapBuffers();   // Swap the front and back buffers every frame
}

/* Function to create shader programs */
void UCreateShader(void) {

	// Specular

	// Specular vertex shader
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &specularVertexShader, NULL);
    glCompileShader(vertexShader);

    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Specular fragment shader
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &specularFragmentShader, NULL);
    glCompileShader(fragmentShader);

    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    specularShader = glCreateProgram();
    glAttachShader(specularShader, vertexShader);
    glAttachShader(specularShader, fragmentShader);
    glLinkProgram(specularShader);

    // check for linking errors
    glGetProgramiv(specularShader, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(specularShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

	// Diffuse

    // Diffuse vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &diffuseVertexShader, NULL);
    glCompileShader(vertexShader);

    // check for shader compile errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Diffuse fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &diffuseFragmentShader, NULL);
    glCompileShader(fragmentShader);

    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    diffuseShader = glCreateProgram();
    glAttachShader(diffuseShader, vertexShader);
    glAttachShader(diffuseShader, fragmentShader);
    glLinkProgram(diffuseShader);

    // check for linking errors
    glGetProgramiv(diffuseShader, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(diffuseShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Lamp

	// Lamp vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &lampVertexShader, NULL);
	glCompileShader(vertexShader);

	// check for shader compile errors
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 51, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Lamp fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &lampFragmentShader, NULL);
	glCompileShader(fragmentShader);

	// check for shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 51, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// link shaders
	lampShader = glCreateProgram();
	glAttachShader(lampShader, vertexShader);
	glAttachShader(lampShader, fragmentShader);
	glLinkProgram(lampShader);

	// check for linking errors
	glGetProgramiv(lampShader, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(lampShader, 51, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// Shadow

    // Shadow vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &shadowVertexShader, NULL);
    glCompileShader(vertexShader);

    // check for shader compile errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Shadow fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &shadowFragmentShader, NULL);
    glCompileShader(fragmentShader);

    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    depthShader = glCreateProgram();
    glAttachShader(depthShader, vertexShader);
    glAttachShader(depthShader, fragmentShader);
    glLinkProgram(depthShader);

    // check for linking errors
    glGetProgramiv(depthShader, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(depthShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

	// Debug

    // Debug vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &debugVertexShader, NULL);
    glCompileShader(vertexShader);

    // check for shader compile errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Debug fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &debugFragmentShader, NULL);
    glCompileShader(fragmentShader);

    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    debugDepthShader = glCreateProgram();
    glAttachShader(debugDepthShader, vertexShader);
    glAttachShader(debugDepthShader, fragmentShader);
    glLinkProgram(debugDepthShader);

    // check for linking errors
    glGetProgramiv(debugDepthShader, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(debugDepthShader, 51, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

/* Function to Create Buffer Objects */
/* Function to generate and load textures */
void UCreateBuffers() {

	/* Chair Object */

	ULoadOBJ(ObjectFile); // Load chair object file data into vectors

	float vertices[14*objVertices.size()]; // Array to hold vertices

	// Loop over all vertices that were loaded into objVertices vector from chair.obj
	for (int i = 0; i < (int)objVertices.size(); ++i) {
		vertices[14*i +  0] = objVertices.at(i).x;   // XYZ coordinates
		vertices[14*i +  1] = objVertices.at(i).y;
		vertices[14*i +  2] = objVertices.at(i).z;
		vertices[14*i +  3] = objNormals.at(i).x;    // Normals
		vertices[14*i +  4] = objNormals.at(i).y;
		vertices[14*i +  5] = objNormals.at(i).z;
		vertices[14*i +  6] = objUVs.at(i).x;        // UV texture coordinates
		vertices[14*i +  7] = objUVs.at(i).y;
		vertices[14*i +  8] = objTangents.at(i).x;   // Tangents (used for normal mapping)
		vertices[14*i +  9] = objTangents.at(i).y;
		vertices[14*i + 10] = objTangents.at(i).z;
		vertices[14*i + 11] = objBitangents.at(i).x; // Bitangents (used for normal mapping)
		vertices[14*i + 12] = objBitangents.at(i).y;
		vertices[14*i + 13] = objBitangents.at(i).z;
	}

	// Generate buffer ids
	glGenVertexArrays(1, &VAO[0]);
	glGenBuffers(1, &VBO[0]);

	// Activate the VAO before binding and setting VBOs and VAPs
	glBindVertexArray(VAO[0]);

	// Activate the VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Instruct GPU how to handle vertex data in buffer
	// Parameters: attribute location | coordinates per vertex | data type | normalization | stride | offset
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// Instruct GPU how to handle normal data in buffer
	// Parameters: attribute location | coordinates per vertex | data type | normalization | stride | offset
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	// Instruct GPU how to handle texture coordinates in buffer
	// Parameters: attribute location | coordinates per vertex | data type | normalization | stride | offset
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	// Set attribute pointer 3 to hold tangent data
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(8 * sizeof(GLfloat)));
	glEnableVertexAttribArray(3);

	// Set attribute pointer 4 to hold bitangent data
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(11 * sizeof(GLfloat)));
	glEnableVertexAttribArray(4);

	// Unbind the VAO
	glBindVertexArray(0);


	/* Floor Plane */

	GLfloat fScale = 500.0f;  // Floor scale
	GLfloat fOffset = -0.73f; // Floor offset y

	// Define Floor
	GLfloat floorVertices[] = {
		 fScale,  fOffset,  fScale,   0.0f, 1.0f, 0.0f,   0.0f,   0.0f,    0.0f, 0.0f, -1.0f,   1.0f, 0.0f, 0.0f,
		-fScale,  fOffset, -fScale,   0.0f, 1.0f, 0.0f,   fScale, fScale,  0.0f, 0.0f, -1.0f,   1.0f, 0.0f, 0.0f,
		-fScale,  fOffset,  fScale,   0.0f, 1.0f, 0.0f,   0.0f,   fScale,  0.0f, 0.0f, -1.0f,   1.0f, 0.0f, 0.0f,
		 fScale,  fOffset,  fScale,   0.0f, 1.0f, 0.0f,   0.0f,   0.0f,    0.0f, 0.0f, -1.0f,   1.0f, 0.0f, 0.0f,
		 fScale,  fOffset, -fScale,   0.0f, 1.0f, 0.0f,   fScale, 0.0f,    0.0f, 0.0f, -1.0f,   1.0f, 0.0f, 0.0f,
		-fScale,  fOffset, -fScale,   0.0f, 1.0f, 0.0f,   fScale, fScale,  0.0f, 0.0f, -1.0f,   1.0f, 0.0f, 0.0f
	};

	// Generate buffer ids
	glGenVertexArrays(1, &VAO[1]);
	glGenBuffers(1, &VBO[1]);

	// Activate the VAO before binding and setting VBOs and VAPs
	glBindVertexArray(VAO[1]);

	// Activate the VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(floorVertices), floorVertices, GL_STATIC_DRAW);

	// Instruct GPU how to handle vertex data in buffer
	// Parameters: attribute location | coordinates per vertex | data type | normalization | stride | offset
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// Instruct GPU how to handle normal data in buffer
	// Parameters: attribute location | coordinates per vertex | data type | normalization | stride | offset
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	// Instruct GPU how to handle texture coordinates in buffer
	// Parameters: attribute location | coordinates per vertex | data type | normalization | stride | offset
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	// Set attribute pointer 3 to hold tangent data
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(8 * sizeof(GLfloat)));
	glEnableVertexAttribArray(3);

	// Set attribute pointer 4 to hold bitangent data
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14*sizeof(GLfloat), (GLvoid*)(11 * sizeof(GLfloat)));
	glEnableVertexAttribArray(4);

	// Unbind the VAO
	glBindVertexArray(0);


	/* Sky Cube */

	float cScale = 100.0f;

	GLfloat skyVertices[] = {
	//   X        Y        Z         U     V
		-cScale,  cScale, -cScale,   0.50, 1.000000,
		 cScale,  cScale,  cScale,   0.25, 0.666666,
		 cScale,  cScale, -cScale,   0.50, 0.666666,
		 cScale,  cScale,  cScale,   0.25, 0.666666,
		-cScale, -cScale,  cScale,   0.00, 0.333333,
		 cScale, -cScale,  cScale,   0.25, 0.333333,
		-cScale,  cScale,  cScale,   1.00, 0.666666,
		-cScale, -cScale, -cScale,   0.75, 0.333333,
		-cScale, -cScale,  cScale,   1.00, 0.333333,
		 cScale, -cScale, -cScale,   0.50, 0.333333,
		-cScale, -cScale,  cScale,   0.25, 0.000000,
		-cScale, -cScale, -cScale,   0.50, 0.000000,
		 cScale,  cScale, -cScale,   0.50, 0.666666,
		 cScale, -cScale,  cScale,   0.25, 0.333333,
		 cScale, -cScale, -cScale,   0.50, 0.333333,
		-cScale,  cScale, -cScale,   0.75, 0.666666,
		 cScale, -cScale, -cScale,   0.50, 0.333333,
		-cScale, -cScale, -cScale,   0.75, 0.333333,
		-cScale,  cScale, -cScale,   0.50, 1.000000,
		-cScale,  cScale,  cScale,   0.25, 1.000000,
		 cScale,  cScale,  cScale,   0.25, 0.666666,
		 cScale,  cScale,  cScale,   0.25, 0.666666,
		-cScale,  cScale,  cScale,   0.00, 0.666666,
		-cScale, -cScale,  cScale,   0.00, 0.333333,
		-cScale,  cScale,  cScale,   1.00, 0.666666,
		-cScale,  cScale, -cScale,   0.75, 0.666666,
		-cScale, -cScale, -cScale,   0.75, 0.333333,
		 cScale, -cScale, -cScale,   0.50, 0.333333,
		 cScale, -cScale,  cScale,   0.25, 0.333333,
		-cScale, -cScale,  cScale,   0.25, 0.000000,
		 cScale,  cScale, -cScale,   0.50, 0.666666,
		 cScale,  cScale,  cScale,   0.25, 0.666666,
		 cScale, -cScale,  cScale,   0.25, 0.333333,
		-cScale,  cScale, -cScale,   0.75, 0.666666,
		 cScale,  cScale, -cScale,   0.50, 0.666666,
		 cScale, -cScale, -cScale,   0.50, 0.333333
	};

	// Generate buffer ids
	glGenVertexArrays(1, &VAO[2]);
	glGenBuffers(1, &VBO[2]);

	// Activate the VAO before binding and setting VBOs and VAPs
	glBindVertexArray(VAO[2]);

	// Activate the VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyVertices), skyVertices, GL_STATIC_DRAW);

	// Set attribute pointer 0 to hold Position data
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// Set attribute pointer 1 to hold UV data
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	// Unbind the VAO
	glBindVertexArray(0);


	/* Test Pane in Screen Space */

    float quadVertices[] = {
    //   X       Y      Z      U     V
        -1.00f, -0.25f, 0.0f,  0.0f, 1.0f,
        -1.00f, -1.00f, 0.0f,  0.0f, 0.0f,
        -0.25f, -0.25f, 0.0f,  1.0f, 1.0f,
        -0.25f, -1.00f, 0.0f,  1.0f, 0.0f,
    };

	// Setup VAO
	glGenVertexArrays(1, &VAO[3]);
	glGenBuffers(1, &VBO[3]);
	glBindVertexArray(VAO[3]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[3]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

	glBindVertexArray(0);

}

void UGenerateTexture() {

    // Load textures
    glGenTextures(6, texture);

    int width, height;
    unsigned char* image;

    // Chair texture
	glActiveTexture(GL_TEXTURE0);                                                              // Use GL_TEXTURE0
	glBindTexture(GL_TEXTURE_2D, texture[0]);                                                  // Bind chair texture
	image = SOIL_load_image(TextureFile1, &width, &height, 0, SOIL_LOAD_RGB);                  // Load chair texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); // Define texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);                          // Linear filter when magnifying
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);            // Blend mipmaps when minifying
	glGenerateMipmap(GL_TEXTURE_2D);                                                           // Generates mipmaps for the specified texture object
	SOIL_free_image_data(image);                                                               // Deactivate image after using

	// Chair normal map
	glActiveTexture(GL_TEXTURE1);                                                              // Use GL_TEXTURE1
	glBindTexture(GL_TEXTURE_2D, texture[1]);                                                  // Bind chair normal map
	image = SOIL_load_image(TextureFile2, &width, &height, 0, SOIL_LOAD_RGB);                  // Load chair normal map
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); // Define texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);                          // Linear filter when magnifying
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);            // Blend mipmaps when minifying
	glGenerateMipmap(GL_TEXTURE_2D);                                                           // Generates mipmaps for the specified texture object
	SOIL_free_image_data(image);                                                               // Deactivate image after using

	// Floor texture
	glActiveTexture(GL_TEXTURE2);                                                              // Use GL_TEXTURE2
	glBindTexture(GL_TEXTURE_2D, texture[2]);                                                  // Bind floor texture
	image = SOIL_load_image(TextureFile3, &width, &height, 0, SOIL_LOAD_RGB);                  // Load floor texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); // Define texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);                          // Linear filter when magnifying
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);            // Blend mipmaps when minifying
	glGenerateMipmap(GL_TEXTURE_2D);                                                           // Generates mipmaps for the specified texture object
	SOIL_free_image_data(image);                                                               // Deactivate image after using

	// Floor normal map
	glActiveTexture(GL_TEXTURE3);                                                              // Use GL_TEXTURE3
	glBindTexture(GL_TEXTURE_2D, texture[3]);                                                  // Bind floor normal map
	image = SOIL_load_image(TextureFile4, &width, &height, 0, SOIL_LOAD_RGB);                  // Load floor normal map
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); // Define texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);                          // Linear filter when magnifying
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);            // Blend mipmaps when minifying
	glGenerateMipmap(GL_TEXTURE_2D);                                                           // Generates mipmaps for the specified texture object
	SOIL_free_image_data(image);                                                               // Deactivate image after using

	// Sky texture
	glActiveTexture(GL_TEXTURE4);                                                              // Use GL_TEXTURE4
	glBindTexture(GL_TEXTURE_2D, texture[4]);                                                  // Bind sky texture
	image = SOIL_load_image(TextureFile5, &width, &height, 0, SOIL_LOAD_RGB);                  // Load sky texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); // Define texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);                          // Linear filter when magnifying
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);            // Blend mipmaps when minifying
	glGenerateMipmap(GL_TEXTURE_2D);                                                           // Generates mipmaps for the specified texture object
	SOIL_free_image_data(image);                                                               // Deactivate image after using

    // configure depth map FBO
    // -----------------------
    glGenFramebuffers(1, &depthMapFBO);
    // create depth texture
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, texture[5]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture[5], 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

/* Callback handler for key press */
void UKeyPressed(unsigned char key, GLint x, GLint y) {
	switch (key) {

	// If escape is pressed, exit program
	case 27:
		exit (0);
		break;

	// If 'p' is pressed, change to orthographic view
	case 'p':
		perspective = !perspective;
		break;

	default:
		break;
    }
}

/* Callback handler for key release (not used) */
void UKeyReleased(unsigned char key, GLint x, GLint y) {
}

/* Callback handler for special key press */
void USpecialKeyPressed(int key, int x, int y) {
	// If ALT is pressed change keyModifier flag to true
	if (glutGetModifiers() & GLUT_ACTIVE_ALT) {
		keyModifier = GL_TRUE;
	}
}

/* Callback handler for special key release */
void USpecialKeyReleased(int key, int x, int y) {
	// If ALT is released change keyModifier flag to false
	keyModifier = GL_FALSE;
}

/* Callback handler for mouse motion */
void UMouseMotion(int x, int y) {

	// If ALT is pressed, allow rotation and zooming
	// if (keyModifier) {

		// If left mouse button is pressed allow rotation
		if (orbitCamera) {
			GLfloat dy = (y - mousey) * 360.0 / WINDOW_HEIGHT; // Get change in y position after click
			GLfloat dx = (x - mousex) * 360.0 / WINDOW_WIDTH;  // Get change in x position after click

			// Set vertical rotation amount. Angle between 0 and 90
			cameraRotV -= dy;
			cameraRotV = cameraRotV < 0.001 ? 0.001 : cameraRotV; // Prevent flipping when x = y = 0
			cameraRotV = cameraRotV > 90 ? 90 : cameraRotV;

			// Set horizontal rotation amount. Keep angle between 0 and 360
			cameraRotH += dx;
			cameraRotH = cameraRotH > 360 ? cameraRotH-360 : cameraRotH;
			cameraRotH = cameraRotH < 0 ? cameraRotH+360 : cameraRotH;

		// If right mouse button is pressed allow zooming
		} else if (zoomCamera) {
			GLfloat dz = (y - mousey) * 5.0 / WINDOW_HEIGHT; // Get change in y position after click

			// Set camera distance
			cameraDistance -= dz;
			if (cameraDistance < 0.001)
				cameraDistance = 0.001; // Prevent flipping when x = y = 0
		}

		// Update mouse variables
		mousex = x;
		mousey = y;

		glutPostRedisplay(); // Redraw frame to allow rotation
	// }
}

/* Callback handler for mouse button press */
void UMousePress(int button, int state, int x, int y) {

	// If mouse button is pressed
	if (state == GLUT_DOWN) {
		switch (button) {

		// If left mouse button is pressed
		case GLUT_LEFT_BUTTON:
			orbitCamera = GL_TRUE;                // Set orbitCamera flag to true
			UMouseMotion(mousex = x, mousey = y); // Call UMouseMotion to orbit
			break;

		// If right mouse button is pressed, zoom into object
		case GLUT_RIGHT_BUTTON:
			zoomCamera = GL_TRUE;                 // Set zoomCamera flag to true
			UMouseMotion(mousex = x, mousey = y); // Call UMouseMotion to zoom
			break;

		default:
			break;
		}

	// If mouse button is released
	} else if (state == GLUT_UP) {
		switch (button) {

		// If left mouse button is released
		case GLUT_LEFT_BUTTON:
			orbitCamera = GL_FALSE; // Set orbitCamera flag to false
			break;

		// If right mouse button is released
		case GLUT_RIGHT_BUTTON:
			zoomCamera = GL_FALSE; // Set zoomCamera flag to false
			break;

		default:
			break;
		}
	}
}

/* Function to load an obj file into global variables */
// OBJECT MUST BE TRIANGULATED WITH VERTICES, UVS AND NORMALS IN THE FILE
void ULoadOBJ(const char* path)
{
	// Define vectors to hold temporary values
	std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
	std::vector<glm::vec3> temp_vertices;
	std::vector<glm::vec2> temp_uvs;
	std::vector<glm::vec3> temp_normals;

	// Open file
	FILE * file = fopen(path, "r");

	// If there is an error opening file
	if( file == NULL ){
		printf("Error opening file\n");
		getchar();
		return;
	}

	// Loop over lines in file
	while( 1 ){

		// Get line
		char lineHeader[128];

		// Read the first word of the line
		int res = fscanf(file, "%s", lineHeader);

		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		// Read vertex coordinates
		if ( strcmp( lineHeader, "v" ) == 0 ){
			glm::vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
			temp_vertices.push_back(vertex);

		// Read vertex texture coordinates (UVs)
		} else if ( strcmp( lineHeader, "vt" ) == 0 ){
			glm::vec2 uv;
			fscanf(file, "%f %f\n", &uv.x, &uv.y );
			temp_uvs.push_back(uv);

		// Read vertex normal coordiantes (normals)
		} else if ( strcmp( lineHeader, "vn" ) == 0 ){
			glm::vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
			temp_normals.push_back(normal);

		// Read face indices
		} else if ( strcmp( lineHeader, "f" ) == 0 ){
			std::string vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );

			// If indices are not of the form vt/vn/f vt/vn/f vt/vn/f
			if (matches != 9){
				printf("Error reading file\n");
				fclose(file);
				return;
			}

			// Append indices to end of temporary vectors
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices    .push_back(uvIndex[0]);
			uvIndices    .push_back(uvIndex[1]);
			uvIndices    .push_back(uvIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);

		} else {

			// Must be a comment - skip
			char temp[1000];
			fgets(temp, 1000, file);
		}
	}

	// For each vertex of each triangle
	for( unsigned int i=0; i<vertexIndices.size(); i++ ){

		// Get indices
		unsigned int vertexIndex = vertexIndices[i];
		unsigned int uvIndex = uvIndices[i];
		unsigned int normalIndex = normalIndices[i];

		// Get the attributes using index
		glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];
		glm::vec2 uv = temp_uvs[ uvIndex-1 ];
		glm::vec3 normal = temp_normals[ normalIndex-1 ];

		// Put attributes in buffers
		objVertices.push_back(vertex);
		objUVs     .push_back(uv);
		objNormals .push_back(normal);
	}

	// Generate tangents and bitangents
	for (unsigned int i=0; i<objVertices.size(); i+=3 ) {

		// Shortcuts for objVertices
		glm::vec3 & v0 = objVertices[i+0];
		glm::vec3 & v1 = objVertices[i+1];
		glm::vec3 & v2 = objVertices[i+2];

		// Shortcuts for objUVs
		glm::vec2 & uv0 = objUVs[i+0];
		glm::vec2 & uv1 = objUVs[i+1];
		glm::vec2 & uv2 = objUVs[i+2];

		// Edges of the triangle : postion delta
		glm::vec3 deltaPos1 = v1-v0;
		glm::vec3 deltaPos2 = v2-v0;

		// UV delta
		glm::vec2 deltaUV1 = uv1-uv0;
		glm::vec2 deltaUV2 = uv2-uv0;

		// Compute tangent and bitangent
		float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
		glm::vec3 tangent   = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * f;
		glm::vec3 bitangent = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * f;

		// Set the same tangent for all three objVertices of the triangle.
		objTangents.push_back(tangent);
		objTangents.push_back(tangent);
		objTangents.push_back(tangent);

		// Same thing for biobjNormals
		objBitangents.push_back(bitangent);
		objBitangents.push_back(bitangent);
		objBitangents.push_back(bitangent);
	}

	// Use Gram-Schmidt orthogonalize to keep normal, bitangent and tangent orthogonal
	for (unsigned int i=0; i<objVertices.size(); i+=1 ) {
		glm::vec3 & n = objNormals[i];
		glm::vec3 & t = objTangents[i];
		glm::vec3 & b = objBitangents[i];

		// Gram-Schmidt orthogonalize
		t = glm::normalize(t - n * glm::dot(n, t));

		// Recompute bitangent
		b = glm::cross(n,t);

		// Calculate handedness
		if (glm::dot(glm::cross(n, t), b) < 0.0f){
			t = t * -1.0f;
		}
	}

	// Close file
	fclose(file);
	return;
}
