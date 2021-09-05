const { mat4, mat3, vec2, vec3, vec4, quat } = glMatrix;

let GL = null;

const _OPAQUE_VS = `#version 300 es
precision highp float;


uniform mat3 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec4 colour;
in vec2 uv0;

out vec2 vUV0;
out vec4 vColour;

void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vColour = colour;
  vUV0 = uv0;
}
`;


const _OPAQUE_FS = `#version 300 es
precision highp float;


uniform sampler2D diffuseTexture;
uniform sampler2D normalTexture;
uniform sampler2D gBuffer_Light;
uniform vec4 resolution;

in vec4 vColour;
in vec2 vUV0;


layout(location = 0) out vec4 out_FragColour;


void main(void) {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 lightSample = texture(gBuffer_Light, uv);
  vec4 albedo = texture(diffuseTexture, vUV0);

  out_FragColour = (albedo * vec4(lightSample.xyz, 1.0) +
      lightSample.w * vec4(0.3, 0.6, 0.1, 0.0));
}
`;


const _QUAD_VS = `#version 300 es
precision highp float;


uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec2 uv0;


void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;


const _QUAD_FS = `#version 300 es
precision highp float;


uniform sampler2D gBuffer_Normal;
uniform sampler2D gBuffer_Position;

uniform vec3 lightColour;

#define _LIGHT_TYPE_POINT

#ifdef _LIGHT_TYPE_DIRECTIONAL
uniform vec3 lightDirection;
#endif

uniform vec3 lightPosition;
uniform vec3 lightAttenuation;

uniform vec3 cameraPosition;
uniform vec4 resolution;


out vec4 out_FragColour;

#define saturate(a) clamp(a, 0.0, 1.0)

float _SmootherStep(float x, float a, float b) {
  x = x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
  return x * (b - a) + a;
}


vec2 _CalculatePhong(vec3 lightDirection, vec3 cameraPosition, vec3 position, vec3 normal) {
  vec3 viewDirection = normalize(cameraPosition - position);
  vec3 H = normalize(lightDirection.xyz + viewDirection);
  float NdotH = dot(normal.xyz, H);
  float specular = saturate(pow(NdotH, 32.0));
  float diffuse = saturate(dot(lightDirection.xyz, normal.xyz));

  return vec2(diffuse, diffuse * specular);
}

vec4 _CalculateLight_Directional(
    vec3 lightDirection, vec3 lightColour, vec3 position, vec3 normal) {

  vec2 lightSample = _CalculatePhong(-lightDirection, cameraPosition, position, normal);

  return vec4(lightSample.x * lightColour, lightSample.y);
}

vec4 _CalculateLight_Point(
    vec3 lightPosition, vec3 lightAttenuation, vec3 lightColour, vec3 position, vec3 normal) {

  vec3 dirToLight = lightPosition - position;
  float lightDistance = length(dirToLight);
  dirToLight = normalize(dirToLight);

  vec2 lightSample = _CalculatePhong(dirToLight, cameraPosition, position, normal);
  float falloff = saturate((lightDistance - lightAttenuation.x) / lightAttenuation.y);

  lightSample *= _SmootherStep(falloff, 1.0, 0.0);

  return vec4(lightSample.x * lightColour, lightSample.y);
}


void main(void) {
  vec2 uv = gl_FragCoord.xy / resolution.xy;

  vec4 normal = texture(gBuffer_Normal, uv);
  vec4 position = texture(gBuffer_Position, uv);

#ifdef _LIGHT_TYPE_DIRECTIONAL
  vec4 lightSample = _CalculateLight_Directional(
      lightDirection, lightColour, position.xyz, normal.xyz);
#elif defined(_LIGHT_TYPE_POINT)
  vec4 lightSample = _CalculateLight_Point(
      lightPosition, lightAttenuation, lightColour, position.xyz, normal.xyz);
#endif

  out_FragColour = lightSample;
}
`;

const _QUAD_COLOUR_VS = `#version 300 es
precision highp float;


uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec2 uv0;


void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;


const _QUAD_COLOUR_FS = `#version 300 es
precision highp float;


uniform sampler2D gQuadTexture;
uniform vec4 resolution;

out vec4 out_FragColour;


void main(void) {
  vec2 uv = gl_FragCoord.xy / resolution.xy;

  out_FragColour = texture(gQuadTexture, uv);
}
`;

const _SIMPLE_VS = `#version 300 es
precision highp float;


uniform mat3 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec2 uv0;

out vec4 vWSPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vUV0;

void main(void) {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vTangent = normalize(normalMatrix * tangent);
  vWSPosition = modelMatrix * vec4(position, 1.0);
  vUV0 = uv0;
}
`;


const _SIMPLE_FS = `#version 300 es
precision highp float;


uniform sampler2D normalTexture;

in vec4 vWSPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vUV0;

layout(location = 0) out vec4 out_Normals;
layout(location = 1) out vec4 out_Position;


void main(void) {
  vec3 bitangent = normalize(cross(vTangent, vNormal));
  mat3 tbn = mat3(vTangent, bitangent, vNormal);
  vec3 normalSample = normalize(texture(normalTexture, vUV0).xyz * 2.0 - 1.0);
  vec3 vsNormal = normalize(tbn * normalSample);

  out_Normals = vec4(vsNormal, 1.0);
  out_Position = vWSPosition;
}
`;

class Shader {
  constructor(vsrc, fsrc, defines) {
    defines = defines || [];

    this._Init(vsrc, fsrc, defines);
  }

  _Init(vsrc, fsrc, defines) {
    this._defines = defines;

    vsrc = this._ModifySourceWithDefines(vsrc, defines);
    fsrc = this._ModifySourceWithDefines(fsrc, defines);

    this._vsSource = vsrc;
    this._fsSource = fsrc;

    this._vsProgram = this._Load(GL.VERTEX_SHADER, vsrc);
    this._fsProgram = this._Load(GL.FRAGMENT_SHADER, fsrc);

    this._shader = GL.createProgram();
    GL.attachShader(this._shader, this._vsProgram);
    GL.attachShader(this._shader, this._fsProgram);
    GL.linkProgram(this._shader);

    if (!GL.getProgramParameter(this._shader, GL.LINK_STATUS)) {
      return null;
    }

    this.attribs = {
      positions: GL.getAttribLocation(this._shader, 'position'),
      normals: GL.getAttribLocation(this._shader, 'normal'),
      tangents: GL.getAttribLocation(this._shader, 'tangent'),
      uvs: GL.getAttribLocation(this._shader, 'uv0'),
      colours: GL.getAttribLocation(this._shader, 'colour'),
    };
    this.uniforms = {
      projectionMatrix: {
        type: 'mat4',
        location: GL.getUniformLocation(this._shader, 'projectionMatrix')
      },
      modelViewMatrix: {
        type: 'mat4',
        location: GL.getUniformLocation(this._shader, 'modelViewMatrix'),
      },
      modelMatrix: {
        type: 'mat4',
        location: GL.getUniformLocation(this._shader, 'modelMatrix'),
      },
      normalMatrix: {
        type: 'mat3',
        location: GL.getUniformLocation(this._shader, 'normalMatrix'),
      },
      resolution: {
        type: 'vec4',
        location: GL.getUniformLocation(this._shader, 'resolution'),
      },
      lightColour: {
        type: 'vec3',
        location: GL.getUniformLocation(this._shader, 'lightColour'),
      },
      lightDirection: {
        type: 'vec3',
        location: GL.getUniformLocation(this._shader, 'lightDirection'),
      },
      lightPosition: {
        type: 'vec3',
        location: GL.getUniformLocation(this._shader, 'lightPosition'),
      },
      lightAttenuation: {
        type: 'vec3',
        location: GL.getUniformLocation(this._shader, 'lightAttenuation'),
      },
      cameraPosition: {
        type: 'vec3',
        location: GL.getUniformLocation(this._shader, 'cameraPosition'),
      },
      diffuseTexture: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'diffuseTexture'),
      },
      normalTexture: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'normalTexture'),
      },
      gBuffer_Light: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'gBuffer_Light'),
      },
      gBuffer_Colour: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'gBuffer_Colour'),
      },
      gBuffer_Normal: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'gBuffer_Normal'),
      },
      gBuffer_Position: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'gBuffer_Position'),
      },
      gQuadTexture: {
        type: 'texture',
        location: GL.getUniformLocation(this._shader, 'gQuadTexture'),
      }
    };
  }

  _ModifySourceWithDefines(src, defines) {
    const lines = src.split('\n');

    const defineStrings = defines.map(d => '#define ' + d);

    lines.splice(3, 0, defineStrings);

    return lines.join('\n');
  }

  _Load(type, source) {
    const shader = GL.createShader(type);

    GL.shaderSource(shader, source);
    GL.compileShader(shader);

    if (!GL.getShaderParameter(shader, GL.COMPILE_STATUS)) {
      console.log(GL.getShaderInfoLog(shader));
      console.log(source);
      GL.deleteShader(shader);
      return null;
    }

    return shader;
  }

  Bind() {
    GL.useProgram(this._shader);
  }
}


class ShaderInstance {
  constructor(shader) {
    this._shaderData = shader;
    this._uniforms = {};
    for (let k in shader.uniforms) {
      this._uniforms[k] = {
        location: shader.uniforms[k].location,
        type: shader.uniforms[k].type,
        value: null
      };
    }
    this._attribs = {...shader.attribs};
  }

  SetMat4(name, m) {
    this._uniforms[name].value = m;
  }

  SetMat3(name, m) {
    this._uniforms[name].value = m;
  }

  SetVec4(name, v) {
    this._uniforms[name].value = v;
  }

  SetVec3(name, v) {
    this._uniforms[name].value = v;
  }

  SetTexture(name, t) {
    this._uniforms[name].value = t;
  }

  Bind(constants) {
    this._shaderData.Bind();

    let textureIndex = 0;

    for (let k in this._uniforms) {
      const v = this._uniforms[k];

      let value = constants[k];
      if (v.value) {
        value = v.value;
      }

      if (value && v.location) {
        const t = v.type;

        if (t == 'mat4') {
          GL.uniformMatrix4fv(v.location, false, value);
        } else if (t == 'mat3') {
          GL.uniformMatrix3fv(v.location, false, value);
        } else if (t == 'vec4') {
          GL.uniform4fv(v.location, value);
        } else if (t == 'vec3') {
          GL.uniform3fv(v.location, value);
        } else if (t == 'texture') {
          value.Bind(textureIndex);
          GL.uniform1i(v.location, textureIndex);
          textureIndex++;
        }
      }
    }
  }
}


class Texture {
  constructor() {
  }

  Load(src) {
    this._name = src;
    this._Load(src);
    return this;
  }

  _Load(src) {
    this._texture = GL.createTexture();
    GL.bindTexture(GL.TEXTURE_2D, this._texture);
    GL.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA,
                  1, 1, 0, GL.RGBA, GL.UNSIGNED_BYTE,
                  new Uint8Array([0, 0, 255, 255]));

    const img = new Image();
    img.src = src;
    img.onload = () => {
      GL.bindTexture(GL.TEXTURE_2D, this._texture);
      GL.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, GL.RGBA, GL.UNSIGNED_BYTE, img);
      GL.generateMipmap(GL.TEXTURE_2D);
      GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.LINEAR_MIPMAP_LINEAR);
      GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.LINEAR);
      GL.bindTexture(GL.TEXTURE_2D, null);
    };
  }

  Bind(index) {
    if (!this._texture) {
      return;
    }
    GL.activeTexture(GL.TEXTURE0 + index);
    GL.bindTexture(GL.TEXTURE_2D, this._texture);
  }

  Unbind() {
    GL.bindTexture(GL.TEXTURE_2D, null);
  }
}


class Mesh {
  constructor() {
    this._buffers = {};

    this._OnInit();
  }

  _BufferData(info, name) {
    if (name == 'index') {
      info.buffer = GL.createBuffer();
      GL.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, info.buffer);
      GL.bufferData(GL.ELEMENT_ARRAY_BUFFER, new Uint16Array(info.data), GL.STATIC_DRAW);
    } else {
      info.buffer = GL.createBuffer();
      GL.bindBuffer(GL.ARRAY_BUFFER, info.buffer);
      GL.bufferData(GL.ARRAY_BUFFER, new Float32Array(info.data), GL.STATIC_DRAW);
    }

    this._buffers[name] = info;
  }

  Bind(shader) {
    for (let k in this._buffers) {
      if (shader._attribs[k] == -1) {
        continue;
      }

      const b = this._buffers[k];

      if (k == 'index') {
        GL.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, b.buffer);
      } else {
        GL.bindBuffer(GL.ARRAY_BUFFER, b.buffer);
        GL.vertexAttribPointer(shader._attribs[k], b.size, GL.FLOAT, false, 0, 0);
        GL.enableVertexAttribArray(shader._attribs[k]);
      }
    }
  }

  Draw() {
    const vertexCount = this._buffers.index.data.length;
    GL.drawElements(GL.TRIANGLES, vertexCount, GL.UNSIGNED_SHORT, 0);
  }
}


class MeshInstance {
  constructor(mesh, shaders, shaderParams) {
    this._mesh = mesh;
    this._shaders = shaders;

    shaderParams = shaderParams || {};
    for (let sk in shaders) {
      const s = shaders[sk];
      for (let k in shaderParams) {
        s.SetTexture(k, shaderParams[k]);
      }
    }

    this._position = vec3.create();
    this._scale = vec3.fromValues(1, 1, 1);
    this._rotation = quat.create();
  }

  SetPosition(x, y, z) {
    vec3.set(this._position, x, y, z);
  }

  RotateX(rad) {
    quat.rotateX(this._rotation, this._rotation, rad);
  }

  RotateY(rad) {
    quat.rotateY(this._rotation, this._rotation, rad);
  }

  Scale(x, y, z) {
    vec3.set(this._scale, x, y, z);
  }

  Bind(constants, pass) {
    const modelMatrix = mat4.create();
    mat4.fromRotationTranslationScale(
        modelMatrix, this._rotation, this._position, this._scale);

    // TODO View matrix
    const viewMatrix = constants['viewMatrix'];
    const modelViewMatrix = mat4.create();
    mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);

    const normalMatrix = mat3.create();
    mat3.fromMat4(normalMatrix, modelMatrix);
    mat3.invert(normalMatrix, normalMatrix);
    mat3.transpose(normalMatrix, normalMatrix);

    const s = this._shaders[pass];

    s.SetMat4('modelViewMatrix', modelViewMatrix);
    s.SetMat4('modelMatrix', modelMatrix);
    s.SetMat3('normalMatrix', normalMatrix);
    s.Bind(constants);

    this._mesh.Bind(s);
  }

  Draw() {
    this._mesh.Draw();
  }
}


class Sphere extends Mesh {
  constructor() {
    super();
  }

  _OnInit() {
    var positions = [];
    for (var i = 0; i < 6; i++) {
        positions = positions.concat([1.0, 0.0, 0.0]);
        positions = positions.concat([1.0, 0.0, 0.0]);
        positions = positions.concat([0.5, 0.8 * Math.sin( (Math.PI/3) * i ), 0.8 * Math.cos( (Math.PI/3) * i)]);
        positions = positions.concat([0.5, 0.8 * Math.sin( (Math.PI/3) * (i-1) ), 0.8 * Math.cos( (Math.PI/3) * (i -1))]);
    }
    for (var i = 0; i < 6; i++) {
        positions = positions.concat([0.5, 0.8 * Math.sin( (Math.PI/3) * i ), 0.8 * Math.cos( (Math.PI/3) * i)]);
        positions = positions.concat([0.5, 0.8 * Math.sin( (Math.PI/3) * (i-1) ), 0.8 * Math.cos( (Math.PI/3) * (i -1))]);
        positions = positions.concat([0.0, Math.sin( (Math.PI/3) * (i-1) ), Math.cos( (Math.PI/3) * (i -1))]);
        positions = positions.concat([0.0, Math.sin( (Math.PI/3) * i ), Math.cos( (Math.PI/3) * i)]);
    }
    for (var i = 0; i < 6; i++) {
        positions = positions.concat([-0.5, 0.8 * Math.sin( (Math.PI/3) * i ), 0.8 * Math.cos( (Math.PI/3) * i)]);
        positions = positions.concat([-0.5, 0.8 * Math.sin( (Math.PI/3) * (i-1) ), 0.8 * Math.cos( (Math.PI/3) * (i -1))]);
        positions = positions.concat([0.0, Math.sin( (Math.PI/3) * (i-1) ), Math.cos( (Math.PI/3) * (i -1))]);
        positions = positions.concat([0.0, Math.sin( (Math.PI/3) * i ), Math.cos( (Math.PI/3) * i)]);
    }
    for (var i = 0; i < 6; i++) {
        positions = positions.concat([-1.0, 0.0, 0.0]);
        positions = positions.concat([-1.0, 0.0, 0.0]);
        positions = positions.concat([-0.5, -0.8 * Math.sin( (Math.PI/3) * i ), -0.8 * Math.cos( (Math.PI/3) * i)]);
        positions = positions.concat([-0.5, -0.8 * Math.sin( (Math.PI/3) * (i-1) ), -0.8 * Math.cos( (Math.PI/3) * (i -1))]);
    }
      /*
    const positions = [
      // Front face
      -1.0, -1.0,  1.0,
      1.0, -1.0,  1.0,
      1.0,  1.0,  1.0,
      -1.0,  1.0,  1.0,

      // Back face
      -1.0, -1.0, -1.0,
      -1.0,  1.0, -1.0,
      1.0,  1.0, -1.0,
      1.0, -1.0, -1.0,

      // Top face
      -1.0,  1.0, -1.0,
      -1.0,  1.0,  1.0,
      1.0,  1.0,  1.0,
      1.0,  1.0, -1.0,

      // Bottom face
      -1.0, -1.0, -1.0,
      1.0, -1.0, -1.0,
      1.0, -1.0,  1.0,
      -1.0, -1.0,  1.0,

      // Right face
      1.0, -1.0, -1.0,
      1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,
      1.0, -1.0,  1.0,

      // Left face
      -1.0, -1.0, -1.0,
      -1.0, -1.0,  1.0,
      -1.0,  1.0,  1.0,
      -1.0,  1.0, -1.0,
    ];
    */

    const uvs = [
      // Front face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Back face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Top face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Bottom face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Right face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Left face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Front face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Back face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Top face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Bottom face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Right face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Left face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      // Front face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Back face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Top face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Bottom face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Right face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Left face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      // Front face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Back face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Top face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Bottom face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Right face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Left face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
    ];

    const normals = [
      // Front face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Back face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,

      // Top face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Bottom face
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,

      // Right face
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,

      // Left face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Front face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Back face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,

      // Top face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Bottom face
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,

      // Right face
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,

      // Left face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Front face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Back face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,

      // Top face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Bottom face
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,

      // Right face
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,

      // Left face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      // Front face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Back face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,

      // Top face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Bottom face
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,

      // Right face
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,

      // Left face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
    ];

    const tangents = [


      // Front face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Back face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Top face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Bottom face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Right face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Left face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      // Front face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Back face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Top face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Bottom face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Right face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Left face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      // Front face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Back face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Top face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Bottom face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Right face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Left face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      // Front face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Back face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Top face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Bottom face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Right face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Left face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
    ];

    /*
    const faceColors = [
      [1.0,  1.0,  1.0,  1.0],    // Front face: white
      [1.0,  0.0,  0.0,  1.0],    // Back face: red
      [0.0,  1.0,  0.0,  1.0],    // Top face: green
      [0.0,  0.0,  1.0,  1.0],    // Bottom face: blue
      [1.0,  1.0,  0.0,  1.0],    // Right face: yellow
      [1.0,  0.0,  1.0,  1.0],    // Left face: purple
    ];
    */

    const faceColors = [
      [1.0,  1.0,  1.0,  1.0],    // Front face: white
      [1.0,  1.0,  1.0,  1.0],    // Back face: red
      [1.0,  1.0,  1.0,  1.0],    // Top face: green
      [1.0,  1.0,  1.0,  1.0],    // Bottom face: blue
      [1.0,  1.0,  1.0,  1.0],    // Right face: yellow
      [1.0,  1.0,  1.0,  1.0],    // Left face: purple
    ];

    // Convert the array of colors into a table for all the vertices.

    let colours = [];

    for (var j = 0; j < faceColors.length; ++j) {
      const c = faceColors[j];

      // Repeat each color four times for the four vertices of the face
      colours = colours.concat(c, c, c, c);
    }

    var indices = [];
    for (var i = 0; i < 24; i++) {
      indices = indices.concat([0 + 4*i, 1 + 4*i, 2 + 4*i,    0 + 4*i, 2 + 4*i, 3 + 4*i]);
    }
    /*
    const indices = [
      0,  1,  2,      0,  2,  3,    // front
      4,  5,  6,      4,  6,  7,    // back
      8,  9,  10,     8,  10, 11,   // top
      12, 13, 14,     12, 14, 15,   // bottom
      16, 17, 18,     16, 18, 19,   // right
      20, 21, 22,     20, 22, 23,   // left

      24, 25, 26,     24, 26, 27,   // other side of sphere
      28, 29, 30,     28, 30, 31,   // other side of sphere
    ];
    */

    this._BufferData({size: 3, data: positions}, 'positions');
    this._BufferData({size: 3, data: normals}, 'normals');
    this._BufferData({size: 3, data: tangents}, 'tangents');
    this._BufferData({size: 4, data: colours}, 'colours');
    this._BufferData({size: 2, data: uvs}, 'uvs');
    this._BufferData({data: indices}, 'index');
  }
}


class Box extends Mesh {
  constructor() {
    super();
  }

  _OnInit() {
    const positions = [
      // Front face
      -1.0, -1.0,  1.0,
      1.0, -1.0,  1.0,
      1.0,  1.0,  1.0,
      -1.0,  1.0,  1.0,

      // Back face
      -1.0, -1.0, -1.0,
      -1.0,  1.0, -1.0,
      1.0,  1.0, -1.0,
      1.0, -1.0, -1.0,

      // Top face
      -1.0,  1.0, -1.0,
      -1.0,  1.0,  1.0,
      1.0,  1.0,  1.0,
      1.0,  1.0, -1.0,

      // Bottom face
      -1.0, -1.0, -1.0,
      1.0, -1.0, -1.0,
      1.0, -1.0,  1.0,
      -1.0, -1.0,  1.0,

      // Right face
      1.0, -1.0, -1.0,
      1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,
      1.0, -1.0,  1.0,

      // Left face
      -1.0, -1.0, -1.0,
      -1.0, -1.0,  1.0,
      -1.0,  1.0,  1.0,
      -1.0,  1.0, -1.0,
    ];

    const uvs = [
      // Front face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Back face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Top face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Bottom face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Right face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,

      // Left face
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
    ];

    const normals = [
      // Front face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Back face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,

      // Top face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Bottom face
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,
      0.0, -1.0, 0.0,

      // Right face
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,

      // Left face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
    ];

    const tangents = [
      // Front face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Back face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Top face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      // Bottom face
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,

      // Right face
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      // Left face
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, -1.0,
    ];

    /*
    const faceColors = [
      [1.0,  1.0,  1.0,  1.0],    // Front face: white
      [1.0,  0.0,  0.0,  1.0],    // Back face: red
      [0.0,  1.0,  0.0,  1.0],    // Top face: green
      [0.0,  0.0,  1.0,  1.0],    // Bottom face: blue
      [1.0,  1.0,  0.0,  1.0],    // Right face: yellow
      [1.0,  0.0,  1.0,  1.0],    // Left face: purple
    ];
    */

    const faceColors = [
      [1.0,  1.0,  1.0,  1.0],    // Front face: white
      [1.0,  1.0,  1.0,  1.0],    // Back face: red
      [1.0,  1.0,  1.0,  1.0],    // Top face: green
      [1.0,  1.0,  1.0,  1.0],    // Bottom face: blue
      [1.0,  1.0,  1.0,  1.0],    // Right face: yellow
      [1.0,  1.0,  1.0,  1.0],    // Left face: purple
    ];

    // Convert the array of colors into a table for all the vertices.

    let colours = [];

    for (var j = 0; j < faceColors.length; ++j) {
      const c = faceColors[j];

      // Repeat each color four times for the four vertices of the face
      colours = colours.concat(c, c, c, c);
    }

    const indices = [
      0,  1,  2,      0,  2,  3,    // front
      4,  5,  6,      4,  6,  7,    // back
      8,  9,  10,     8,  10, 11,   // top
      12, 13, 14,     12, 14, 15,   // bottom
      16, 17, 18,     16, 18, 19,   // right
      20, 21, 22,     20, 22, 23,   // left
    ];

    this._BufferData({size: 3, data: positions}, 'positions');
    this._BufferData({size: 3, data: normals}, 'normals');
    this._BufferData({size: 3, data: tangents}, 'tangents');
    this._BufferData({size: 4, data: colours}, 'colours');
    this._BufferData({size: 2, data: uvs}, 'uvs');
    this._BufferData({data: indices}, 'index');
  }
}

class Quad extends Mesh {
  constructor() {
    super();
  }

  _OnInit() {
    const positions = [
      -0.5, -0.5, 1.0,
      0.5, -0.5, 1.0,
      0.5, 0.5, 1.0,
      -0.5, 0.5, 1.0,
    ];

    const normals = [
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
    ];

    const tangents = [
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
      -1.0, 0.0, 0.0,
    ];

    const uvs = [
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
    ];

    const indices = [
      0, 1, 2,
      0, 2, 3,
    ];

    this._BufferData({size: 3, data: positions}, 'positions');
    this._BufferData({size: 3, data: normals}, 'normals');
    this._BufferData({size: 3, data: tangents}, 'tangents');
    this._BufferData({size: 2, data: uvs}, 'uvs');
    this._BufferData({data: indices}, 'index');
  }
}


class Camera {
  constructor() {
    this._position = vec3.create();
    this._target = vec3.create();
    this._viewMatrix = mat4.create();
    this._cameraMatrix = mat4.create();
  }

  SetPosition(x, y, z) {
    vec3.set(this._position, x, y, z);
  }

  SetTarget(x, y, z) {
    vec3.set(this._target, x, y, z);
  }

  UpdateConstants(constants) {
    mat4.lookAt(this._viewMatrix, this._position, this._target, vec3.fromValues(0, 1, 0));
    mat4.invert(this._cameraMatrix, this._viewMatrix);

    constants['projectionMatrix'] = this._projectionMatrix;
    constants['viewMatrix'] = this._viewMatrix;
    constants['cameraMatrix'] = this._cameraMatrix;
    constants['cameraPosition'] = this._position;
  }
}


class PerspectiveCamera extends Camera {
  constructor(fov, aspect, zNear, zFar) {
    super();

    this._projectionMatrix = mat4.create();
    this._fov = fov;
    this._aspect = aspect;
    this._zNear = zNear;
    this._zFar = zFar;

    mat4.perspective(this._projectionMatrix, fov * Math.PI / 180.0, aspect, zNear, zFar);
  }

  GetUp() {
    const v = vec4.fromValues(0, 0, 1, 0);

    vec4.transformMat4(v, v, this._cameraMatrix);

    return v;
  }

  GetRight() {
    const v = vec4.fromValues(1, 0, 0, 0);

    vec4.transformMat4(v, v, this._cameraMatrix);

    return v;
  }
}


class OrthoCamera extends Camera {
  constructor(l, r, b, t, n, f) {
    super();

    this._projectionMatrix = mat4.create();

    mat4.ortho(this._projectionMatrix, l, r, b, t, n, f);
  }
}


class Light {
  constructor() {
  }

  UpdateConstants() {
  }
}


class DirectionalLight extends Light {
  constructor() {
    super();

    this._colour = vec3.fromValues(1, 1, 1);
    this._direction = vec3.fromValues(1, 0, 0);
  }

  get Type() {
    return 'Directional';
  }

  SetColour(r, g, b) {
    vec3.set(this._colour, r, g, b);
  }

  SetDirection(x, y, z) {
    vec3.set(this._direction, x, y, z);
    vec3.normalize(this._direction, this._direction);
  }

  UpdateConstants(constants) {
    constants['lightDirection'] = this._direction;
    constants['lightColour'] = this._colour;
  }
}

class PointLight extends Light {
  constructor() {
    super();

    this._colour = vec3.fromValues(1, 1, 1);
    this._position = vec3.create();
    this._attenuation = vec3.create();
  }

  get Type() {
    return 'Point';
  }

  SetColour(r, g, b) {
    vec3.set(this._colour, r, g, b);
  }

  SetPosition(x, y, z) {
    vec3.set(this._position, x, y, z);
  }

  SetRadius(r1, r2) {
    vec3.set(this._attenuation, r1, r2, 0);
  }

  UpdateConstants(constants) {
    constants['lightPosition'] = this._position;
    constants['lightColour'] = this._colour;
    constants['lightAttenuation'] = this._attenuation;
  }
}

class Renderer {
  constructor() {
    this._Init();
  }

  _Init() {
    this._canvas = document.createElement('canvas');

    document.body.appendChild(this._canvas);

    document.addEventListener('keydown', keyPush);
    document.addEventListener('keyup', keyUp);

    GL = this._canvas.getContext('webgl2');

    if (GL === null) {
      alert("Unable to initialize WebGL. Your browser or machine may not support it.");
      return;
    }

    this._constants = {};

    this._textures = {};
    this._textures['test-diffuse'] = new Texture().Load('./resources/rough-wet-cobble-albedo-1024.png');
    this._textures['test-normal'] = new Texture().Load('./resources/rough-wet-cobble-normal-1024.jpg');
    this._textures['worn-bumpy-rock-albedo'] = new Texture().Load(
        './resources/worn-bumpy-rock-albedo-1024.png');
    this._textures['worn-bumpy-rock-normal'] = new Texture().Load(
        './resources/worn-bumpy-rock-normal-1024.jpg');

    this._shaders = {};
    this._shaders['z'] = new Shader(_SIMPLE_VS, _SIMPLE_FS);
    this._shaders['default'] = new Shader(_OPAQUE_VS, _OPAQUE_FS);

    this._shaders['post-quad-colour'] = new Shader(
        _QUAD_COLOUR_VS, _QUAD_COLOUR_FS);
    this._shaders['post-quad-directional'] = new Shader(
        _QUAD_VS, _QUAD_FS, ['_LIGHT_TYPE_DIRECTIONAL']);
    this._shaders['post-quad-point'] = new Shader(
        _QUAD_VS, _QUAD_FS, ['_LIGHT_TYPE_POINT']);

    this._camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1.0, 1000.0);
    this._camera.SetPosition(0, 20, 10);
    this._camera.SetTarget(0, 0, -20);

    this._postCamera = new OrthoCamera(0.0, 1.0, 0.0, 1.0, 1.0, 1000.0);

    this._meshes = [];
    this._lights = [];

    this._quadDirectional = new MeshInstance(
        new Quad(),
        {light: new ShaderInstance(this._shaders['post-quad-directional'])});
    this._quadDirectional.SetPosition(0.5, 0.5, -10.0);

    this._quadPoint = new MeshInstance(
        new Quad(),
        {light: new ShaderInstance(this._shaders['post-quad-point'])});
    this._quadPoint.SetPosition(0.5, 0.5, -10.0);

    this._quadColour = new MeshInstance(
        new Quad(),
        {colour: new ShaderInstance(this._shaders['post-quad-colour'])});
    this._quadColour.SetPosition(0.5, 0.5, -10.0);

    this._InitGBuffer();
    this.Resize(window.innerWidth, window.innerHeight);
  }

  _InitGBuffer() {
    // Float textures have only been around for like 15 years.
    // So of course make them an extension.
    GL.getExtension('EXT_color_buffer_float');

    this._depthBuffer = GL.createRenderbuffer();
    GL.bindRenderbuffer(GL.RENDERBUFFER, this._depthBuffer);
    GL.renderbufferStorage(
        GL.RENDERBUFFER,
        GL.DEPTH_COMPONENT24,
        window.innerWidth, window.innerHeight);
    GL.bindRenderbuffer(GL.RENDERBUFFER, null);

    this._normalBuffer = GL.createTexture();
    GL.bindTexture(GL.TEXTURE_2D, this._normalBuffer);
    GL.texImage2D(
        GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
        0, GL.RGBA, GL.FLOAT, null);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
      GL.bindTexture(GL.TEXTURE_2D, null);

    this._positionBuffer = GL.createTexture();
    GL.bindTexture(GL.TEXTURE_2D, this._positionBuffer);
    GL.texImage2D(
        GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
        0, GL.RGBA, GL.FLOAT, null);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
    GL.bindTexture(GL.TEXTURE_2D, null);

    this._lightBuffer = GL.createTexture();
    GL.bindTexture(GL.TEXTURE_2D, this._lightBuffer);
    GL.texImage2D(
        GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
        0, GL.RGBA, GL.FLOAT, null);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
    GL.bindTexture(GL.TEXTURE_2D, null);

    this._colourBuffer = GL.createTexture();
    GL.bindTexture(GL.TEXTURE_2D, this._colourBuffer);
    GL.texImage2D(
        GL.TEXTURE_2D, 0, GL.RGBA32F, window.innerWidth, window.innerHeight,
        0, GL.RGBA, GL.FLOAT, null);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
    GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
    GL.bindTexture(GL.TEXTURE_2D, null);

    // Create the FBO's for each pass
    this._zFBO = GL.createFramebuffer();
    GL.bindFramebuffer(GL.FRAMEBUFFER, this._zFBO);
    GL.framebufferRenderbuffer(
        GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, this._depthBuffer);
    GL.framebufferTexture2D(
        GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, this._normalBuffer, 0);
    GL.framebufferTexture2D(
        GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT1, GL.TEXTURE_2D, this._positionBuffer, 0);
    GL.bindFramebuffer(GL.FRAMEBUFFER, null);

    this._lightFBO = GL.createFramebuffer();
    GL.bindFramebuffer(GL.FRAMEBUFFER, this._lightFBO);
    GL.framebufferRenderbuffer(
        GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, this._depthBuffer);
    GL.framebufferTexture2D(
        GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, this._lightBuffer, 0);
    GL.bindFramebuffer(GL.FRAMEBUFFER, null);

    this._colourFBO = GL.createFramebuffer();
    GL.bindFramebuffer(GL.FRAMEBUFFER, this._colourFBO);
    GL.framebufferRenderbuffer(
        GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, this._depthBuffer);
    GL.framebufferTexture2D(
        GL.DRAW_FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, this._colourBuffer, 0);
    GL.bindFramebuffer(GL.FRAMEBUFFER, null);

    // GROSS
    this._normalTexture = new Texture();
    this._normalTexture._texture = this._normalBuffer;

    this._positionTexture = new Texture();
    this._positionTexture._texture = this._positionBuffer;

    this._lightTexture = new Texture();
    this._lightTexture._texture = this._lightBuffer;

    this._colourTexture = new Texture();
    this._colourTexture._texture = this._colourBuffer;
  }

  CreateMeshInstance(mesh, shaderParams) {
    const params = {};
    for (let k in shaderParams.params) {
      params[k] = this._textures[shaderParams.params[k]];
    }

    const m = new MeshInstance(
        mesh,
        {
          z: new ShaderInstance(this._shaders['z']),
          colour: new ShaderInstance(this._shaders[shaderParams.shader])
        }, params);

    this._meshes.push(m);

    return m;
  }

  CreateLight(type) {
    let l = null;

    if (type == 'directional') {
      l = new DirectionalLight();
    } else if (type == 'point') {
      l = new PointLight();
    }

    if (!l) {
      return null;
    }

    this._lights.push(l);

    return l;
  }

  Resize(w, h) {
    this._canvas.width = w;
    this._canvas.height = h;
    GL.viewport(0, 0, w, h);
  }

  _SetQuadSizeForLight(quad, light) {
    const wvp = mat4.create();
    const w = mat4.create();
    mat4.fromTranslation(w, light._position);

    const viewMatrix = this._camera._viewMatrix;
    const projectionMatrix = this._camera._projectionMatrix;

    const _TransformToScreenSpace = (p) => {
      const screenPos = vec4.fromValues(
          p[0], p[1], p[2], 1.0);

      vec4.transformMat4(screenPos, screenPos, projectionMatrix);

      screenPos[0] = (screenPos[0] / screenPos[3]) * 0.5 + 0.5;
      screenPos[1] = (screenPos[1] / screenPos[3]) * 0.5 + 0.5;

      return screenPos;
    };

    const lightRadius = (light._attenuation[0] + light._attenuation[1]);
    const lightDistance = vec3.distance(this._camera._position, light._position);

    if (lightDistance < lightRadius) {
      quad.SetPosition(0.5, 0.5, -10);
      quad.Scale(1, 1, 1);
    } else {
      const viewSpaceCenter = vec3.clone(light._position);
      vec3.transformMat4(viewSpaceCenter, viewSpaceCenter, viewMatrix);

      const rightPos = vec3.clone(viewSpaceCenter);
      const upPos = vec3.clone(viewSpaceCenter);
      vec3.add(rightPos, rightPos, vec3.fromValues(lightRadius, 0, 0));
      vec3.add(upPos, upPos, vec3.fromValues(0, -lightRadius, 0));

      const center = _TransformToScreenSpace(light._position);
      const up = _TransformToScreenSpace(upPos);
      const right = _TransformToScreenSpace(rightPos);

      const radius = 2 * Math.max(
          vec2.distance(center, up), vec2.distance(center, right));

      quad.SetPosition(center[0], center[1], -10);
      quad.Scale(radius, radius, 1);
    }
  }

  Render(timeElapsed) {
    this._constants['resolution'] = vec4.fromValues(
        window.innerWidth, window.innerHeight, 0, 0);
    this._camera.UpdateConstants(this._constants);

    this._constants['gBuffer_Normal'] = null;
    this._constants['gBuffer_Position'] = null;
    this._constants['gBuffer_Colour'] = null;
    this._constants['gBuffer_Light'] = null;

    // Z-Prepass + normals
    GL.bindFramebuffer(GL.FRAMEBUFFER, this._zFBO);
    GL.drawBuffers([GL.COLOR_ATTACHMENT0, GL.COLOR_ATTACHMENT1]);

    GL.clearColor(0.0, 0.0, 0.0, 0.0);
    GL.clearDepth(1.0);
    GL.enable(GL.DEPTH_TEST);
    GL.depthMask(true);
    GL.depthFunc(GL.LEQUAL);
    GL.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);

    this._camera.UpdateConstants(this._constants);

    for (let m of this._meshes) {
      m.Bind(this._constants, 'z');
      m.Draw();
    }

    GL.useProgram(null);
    GL.bindTexture(GL.TEXTURE_2D, null);

    // Light buffer generation
    GL.bindFramebuffer(GL.FRAMEBUFFER, this._lightFBO);
    GL.drawBuffers([GL.COLOR_ATTACHMENT0]);

    GL.clear(GL.COLOR_BUFFER_BIT);
    GL.disable(GL.DEPTH_TEST);
    GL.enable(GL.BLEND);
    GL.blendFunc(GL.ONE, GL.ONE);

    this._postCamera.UpdateConstants(this._constants);

    this._constants['gBuffer_Normal'] = this._normalTexture;
    this._constants['gBuffer_Position'] = this._positionTexture;

    for (let l of this._lights) {
      l.UpdateConstants(this._constants);

      let quad = null;
      if (l.Type == 'Directional') {
        quad = this._quadDirectional;
      } else if (l.Type == 'Point') {
        quad = this._quadPoint;

        // Calculate screenspace size
        this._SetQuadSizeForLight(quad, l);
      }
      quad.Bind(this._constants, 'light');
      quad.Draw();
    }

    GL.useProgram(null);
    GL.bindTexture(GL.TEXTURE_2D, null);

    // Colour pass
    GL.bindFramebuffer(GL.FRAMEBUFFER, this._colourFBO);
    GL.drawBuffers([GL.COLOR_ATTACHMENT0]);
    GL.disable(GL.BLEND);
    GL.depthMask(false);
    GL.enable(GL.DEPTH_TEST);

    this._camera.UpdateConstants(this._constants);

    this._constants['gBuffer_Colour'] = null;
    this._constants['gBuffer_Light'] = this._lightTexture;
    this._constants['gBuffer_Normal'] = null;
    this._constants['gBuffer_Position'] = null;

    for (let m of this._meshes) {
      m.Bind(this._constants, 'colour');
      m.Draw();
    }

    GL.useProgram(null);
    GL.bindTexture(GL.TEXTURE_2D, null);
    GL.disable(GL.BLEND);

    // Now just draw directly to screen
    GL.bindFramebuffer(GL.FRAMEBUFFER, null);
    GL.disable(GL.DEPTH_TEST);
    GL.disable(GL.BLEND);

    this._postCamera.UpdateConstants(this._constants);

    // Really fucking hate JavaScript sometimes.
    this._constants['gQuadTexture'] = this._colourTexture;

    this._quadColour.Bind(this._constants, 'colour');
    this._quadColour.Draw();
  }
}

class LightPrepassDemo {
  constructor() {
    this._Initialize();
  }

  _Initialize() {
    this._renderer = new Renderer();

    window.addEventListener('resize', () => {
      this._OnWindowResize();
    }, false);

    this._Init();

    this._previousRAF = null;
    this._RAF();
  }

  _OnWindowResize() {
    this._renderer.Resize(window.innerWidth, window.innerHeight);
  }

  _Init() {
    this._CreateLights();
    this._CreateMeshes();
  }

  _CreateLights() {
    this._lights = [];

    for (let i = -1; i <= 10; i++) {
      let l = this._renderer.CreateLight('point');

      const v = vec3.fromValues(1.0, 1.0, 1.0);
      vec3.normalize(v, v);

      /*
      const p = vec3.fromValues(
        (Math.random() * 2 - 1) * 10,
        3,
        -Math.random() * 10 - 10);
      */
      const p = vec3.fromValues(
        -5 + 50 * i,
        15,
        30);

      l.SetColour(v[0], v[1], v[2]);
      l.SetPosition(p[0], p[1], p[2]);
      l.SetRadius(100, 1);

      this._lights.push({
          light: l,
          position: p,
          acc: 0, //Math.random() * 10.0,
          accSpeed: 0 //Math.random() * 0.5 + 0.5,
      });
    }
  }

  _CreateMeshes() {
    this._meshes = [];

    let m = this._renderer.CreateMeshInstance(
        new Quad(),
        {
          shader: 'default',
          params: {
            diffuseTexture: 'worn-bumpy-rock-albedo',
            normalTexture: 'worn-bumpy-rock-normal',
          }
        });
    m.SetPosition(0, -2, -10);
    m.RotateX(-Math.PI * 0.5);
    m.Scale(1000, 1000, 1);


    let background = this._renderer.CreateMeshInstance(
        new Quad(),
        {
          shader: 'default',
          params: {
            diffuseTexture: 'test-diffuse',
            normalTexture: 'worn-bumpy-rock-normal',
          }
        });
    background.SetPosition(0, -2, -50);
    //m.RotateX(-Math.PI * 0.5);
    background.Scale(5000, 5000, 1);

    character = this._CreateMovingEntity();

    enemies[0] = this._CreateMovingEntity();

    enemies[0]._startMoveRight();
  }

  _CreateMovingEntity() {
    let x = -1;
    let y = 2;
    var body = this._renderer.CreateMeshInstance(
        new Sphere(),
        {
          shader: 'default',
          params: {
            diffuseTexture: 'test-diffuse',
            normalTexture: 'test-normal',
          }
        });
    body.SetPosition(x * 4, 0, -y * 4);
    this._meshes.push(body);

    var arms = [];
    arms[0] = this._renderer.CreateMeshInstance(
        new Sphere(),
        {
          shader: 'default',
          params: {
            diffuseTexture: 'test-diffuse',
            normalTexture: 'test-normal',
          }
        });
    arms[0].SetPosition(x * 4 + 0.2, 0, -y * 4 + 0.9);
    arms[0].Scale(0.3, 0.3, 0.3);
    this._meshes.push(arms[0]);

    var nose = this._renderer.CreateMeshInstance(
        new Sphere(),
        {
          shader: 'default',
          params: {
            diffuseTexture: 'test-diffuse',
            normalTexture: 'test-normal',
          }
        });
    nose.SetPosition(x * 3.2 + 0.2, 0, -y * 4 + 0.9);
    nose.Scale(0.2, 0.2, 0.2);
    this._meshes.push(nose);

    var feet = [];
    feet[0] = this._renderer.CreateMeshInstance(
        new Sphere(),
        {
          shader: 'default',
          params: {
            diffuseTexture: 'test-diffuse',
            normalTexture: 'test-normal',
          }
        });
    feet[0].SetPosition(x * 4 + 0.2, -0.3, -y * 4 + 0.9);
    feet[0].Scale(0.3, 0.2, 0.3);
    this._meshes.push(feet[0]);

    var inhalingDust = [];
    const box = new Box();
    for (var i = 0; i < 10; i++) {
      var dust = this._renderer.CreateMeshInstance(
        box,
        {
          shader: 'default',
          params: {
            diffuseTexture: 'test-diffuse',
            normalTexture: 'test-normal',
          }
        });
      // move it out of view for now
      dust.SetPosition(5000, 5000, 5000);
      dust.Scale(Math.random() * 0.3, 0.01, Math.random() * 0.3);
      inhalingDust.push(dust);
    }
    return new MovingEntity(body, arms, feet, nose, inhalingDust);
  }

  _RAF() {
    requestAnimationFrame((t) => {
      if (this._previousRAF === null) {
        this._previousRAF = t;
      }

      this._RAF();
      this._Step(t - this._previousRAF);
      this._previousRAF = t;
    });
  }

  _Step(timeElapsed) {
    const timeElapsedS = timeElapsed * 0.001;

    character._Step(timeElapsed);

    for (var i = 0; i < enemies.length; i++) {
      var enemy = enemies[i];
      enemy._Step(timeElapsed);
      var changeAction = Math.random();
      if (changeAction < 0.01) {
        enemy._startMoveLeft();
      } else if (changeAction < 0.02) {
        enemy._startMoveRight();
      } else if (changeAction < 0.03) {
        enemy._jump();
      } else if (changeAction < 0.04) {
        enemy._startInhaling();
      } else if (changeAction < 0.05) {
        enemy._stopInhaling();
      }
    }

    // TODO DELETE
    for (let l of this._lights) {
      l.acc += timeElapsed * 0.001 * l.accSpeed;

      l.light.SetPosition(
          l.position[0] + 10 * Math.cos(l.acc),
          l.position[1],
          l.position[2] + 10 * Math.sin(l.acc));
    }

    //console.log(this);
    //this._camera.SetPosition(0, 20, 10);
    //this._camera.SetTarget(0, 0, -20);
    this._renderer._camera.SetPosition(
        character.body._position[0],
        character.body._position[1] + 3 +
        (character.inhaling ? -inhalingPosShift : (character.ducking ? duckingPosShift : 0)),
        character.body._position[2] + 15);
    this._renderer._camera.SetTarget(
        character.body._position[0],
        character.body._position[1] +
        (character.inhaling ? -inhalingPosShift : (character.ducking ? duckingPosShift : 0)),
        character.body._position[2]);

    this._renderer.Render(timeElapsedS);
  }
}

var enemies = [];
var character;

var groundFriction = 0.05;
var gravity = 0.1;
var flyingGravity = 0.05;
const duckingPosShift = 0.5;
const inhalingPosShift = 1.0;
const inhalingWalkingSpeed = 0.2;
const walkingSpeed = 0.4;
function keyPush(evt) {
  switch (evt.keyCode) {
    case 65: // A  (left in wasd)
      character._startMoveLeft();
      break;
    case 87: // W  (up in wasd)
      break;
    case 68: // D  (right in wasd)
      character._startMoveRight();
      break;
    case 83: // S  (down in wasd)
      break;
    case 74: // J
      character._startInhaling();
      break;
    case 75: // K
      character._jump();
      break;
    case 76: // L
      break;
  }
}

function keyUp(evt) {
  switch (evt.keyCode) {
    case 65: // A  (left in wasd)
      character._stopMoveLeft();
      break;
    case 87: // W  (up in wasd)
      // cannot cancel jump with keyup
      break;
    case 68: // D  (right in wasd)
      character._stopMoveRight();
      break;
    case 83: // S  (down in wasd)
      character._stopDucking();
      break;
    case 74: // K
      character._stopInhaling();
      break;
  }
}

class MovingEntity {
  constructor(bodyMesh, armMeshes, feetMeshes, noseMesh, inhalingDustMeshes) {
    this.body = bodyMesh;

    this.moveLeft = this.moveRight = this.jumping = this.flying = this.ducking = this.inhaling = false;
    // 1 for right, -1 for left
    this.facingDirection = 1;
    this.yVelocity = 0;
    this.xVelocity = 0;
    this.nose = noseMesh;
    this.arms = armMeshes;
    this.feet = feetMeshes;
    this.inhalingDust = inhalingDustMeshes;
    // 2D array, storing x and y offset from neutral in middle of body
    this.armPos = [[0.0, 0.0], [0.0, 0.0]];
    this.armSwing = 0.0;
    // a yVelocity for arms flapping while floating/flying
    this.armFlapFloating = 0.0;
  }

  _Initialize() {
  }

  _startMoveLeft() {
    if (!this.moveLeft && !this.ducking) {
      this.xVelocity = this.inhaling ? -inhalingWalkingSpeed  : -walkingSpeed;
      this.armSwing = 0.05;
      this.moveLeft = true;
      this.facingDirection = -1;
    }
  }

  _stopMoveLeft() {
    this.moveLeft = false;
  }

  _startMoveRight() {
    if (!this.moveRight && !this.ducking) {
      this.xVelocity = this.inhaling ? inhalingWalkingSpeed  : walkingSpeed;
      this.armSwing = 0.05;
      this.moveRight = true;
      this.facingDirection = 1;
    }
  }

  _stopMoveRight() {
    this.moveRight = false;
  }

  _startDucking() {
    if (!this.jumping && !this.flying && !this.ducking && !this.inhaling) {
      this.ducking = true;
      this.moveLeft = this.moveRight = false;
      this.body.Scale(1.0, 0.3, 1.0);
      // TODO - make it so this doesn't impact the camera
      this.body.SetPosition(
        this.body._position[0], this.body._position[1] - duckingPosShift, this.body._position[2]);
    }
  }

  _stopDucking() {
    if (!this.jumping && !this.flying && this.ducking) {
      this.ducking = false;
      this.body.Scale(1.0, 1.0, 1.0);
      this.body.SetPosition(
        this.body._position[0], this.body._position[1] + duckingPosShift, this.body._position[2]);
    }
  }

  _startInhaling() {
    // deflate and go back to original fall speed
    if (this.flying) {
      this.flying = false;
      this.body.Scale(1.0, 1.0, 1.0);
      // TODO DELETE - set back to normal gravity
      //gravity = 0.1;
    } else {
      if (!this.inhaling && !this.ducking) {
        this.inhaling = true;
        // walk slower while inhaling
        if (this.xVelocity !=0) this.xVelocity *= 0.5;
        const charPos = this.body._position;

        // place the dust particles to show inhaling
        for (var i = 0; i < this.inhalingDust.length; i++) {
          this.inhalingDust[i].SetPosition(
            charPos[0] + 1.5 + Math.random() * 3.0,
            charPos[1] + Math.random() * 2.5,
            charPos[2] + Math.random() * 2.5,
          );
          this.inhalingDust[i].RotateX(Math.random());
          this.inhalingDust[i].RotateY(Math.random());
        }

        this.body.Scale(1.2, 1.8, 1.8);
        this.body.SetPosition(
          charPos[0],
          charPos[1] + inhalingPosShift,
          charPos[2]);
      }
    }
  }

  _stopInhaling() {
    if (this.inhaling) {
      this.inhaling = false;
      if (this.moveLeft) this.xVelocity = this.inhaling ? -inhalingWalkingSpeed  : -walkingSpeed;
      if (this.moveRight) this.xVelocity = this.inhaling ? inhalingWalkingSpeed  : walkingSpeed;

      for (var i = 0; i < this.inhalingDust.length; i++) { // move the dust out of view
        var dust = this.inhalingDust[i].SetPosition(500, 500, 500);
      }
      this.body.Scale(1.0, 1.0, 1.0);
      this.body.SetPosition(
        this.body._position[0],
        this.body._position[1] - inhalingPosShift,
        this.body._position[2]);
    }
  }

  _jump() {
      // second jump, puff up and start flying
      if (this.jumping) {
        this.body.Scale(1.8, 1.8, 1.8);
        // TODO DELETE - set floating gravity
        //gravity = 0.05;
        this.yVelocity = 0.7;
        this.armFlapFloating = 0.2;
        this.flying = true;
      } else if (this.ducking) {
        // TODO - sliding kick
      } else if (!this.inhaling) {
        this.yVelocity = 1.5;
        this.jumping = true;
      }
  }

  // TODO - scale the amounts in this function by timeElapsed
  _Step(timeElapsed) {

    let m = this.body;
    const currPos = m._position;
    currPos[0] += this.xVelocity;

    // ground friction
    if (!this.moveLeft && this.xVelocity < 0 ) {
        this.xVelocity += groundFriction;
        if (this.xVelocity > 0) {
          this.armSwing = 0;
          this.xVelocity = 0;
        }
    }
    if (!this.moveRight && this.xVelocity > 0 ) {
      this.xVelocity -= groundFriction;
      if (this.xVelocity < 0) {
        this.armSwing = 0;
        this.xVelocity = 0;
      }
    }

    // TODO - terminal velocity, which should be lower when puffed up
    if (this.yVelocity != 0) {
        currPos[1] += this.yVelocity;
        this.yVelocity -= this.flying ? flyingGravity : gravity;
    }
    // hit the ground after jumping?
    if (this.yVelocity != 0 && currPos[1] < 0) {
        this.yVelocity = 0;
        // if inhaling we want to stay big
        if (this.inhaling) {
            currPos[1] = inhalingPosShift;
        } else {
            this.body.Scale(1.0, 1.0, 1.0);
            currPos[1] = 0;
        }
        this.jumping = false;
        this.flying = false;
    }

    if ( !this.flying && !this.jumping && (this.moveLeft || this.moveRight)) {
      this.armPos[0][0] += this.armSwing;

      if (this.armPos[0][0] > 0.2) {
        this.armSwing = -0.05
      }
      else if (this.armPos[0][0] < -0.2) {
        this.armSwing = 0.05
      }
    }

    if (this.flying) {

      if (this.armPos[0][1] > 0.3 && this.armFlapFloating > 0) {
        this.armFlapFloating = -0.01;
      }
      this.armPos[0][1] += this.armFlapFloating;
      // make arms fly backwards while they flap up
      this.armPos[0][0] = -1 * this.facingDirection * 0.2;
      if (this.armFlapFloating !== 0) {
        this.armFlapFloating -= gravity * 0.1;
      }
      //console.log(armPos, currPos);
      if (this.armPos[0][1] < -0.2) {
          this.armFlapFloating = 0;
      }
    }

    // arms animation
    this.arms[0].SetPosition(
        currPos[0] + 0.0 + this.armPos[0][0],
        currPos[1] + 0.3 + this.armPos[0][1],
        currPos[2] +
        (this.flying || this.inhaling ? 1.8 : 0.9)
    );

    this.nose.SetPosition(
        currPos[0] +
        this.facingDirection *
        (this.flying ? 1.5 : (this.inhaling ? 1.1 : 0.9)),
        currPos[1] + 0.2,
        currPos[2] +
        (this.flying || this.inhaling ? 1.2 : 0.9)
    );

    this.feet[0].SetPosition(
        currPos[0] + 0.0 - this.armPos[0][0],
        currPos[1]
          - (this.flying || this.inhaling ? 1.2 : 0.6),
        currPos[2] +
        (this.flying || this.inhaling ? 1.8 : (this.ducking ? 0.1 : 0.9))
    );

    m.SetPosition(currPos[0], currPos[1], currPos[2]);

    // move the inhaling dust
    if (this.inhaling) {
      for (var i = 0; i < this.inhalingDust.length; i++) {
        var dust = this.inhalingDust[i];
        var dPos = dust._position;
        dust.SetPosition(
          dPos[0] - this.facingDirection * 0.3,
          dPos[1] - ((dPos[1] - currPos[1]) * .1),
          dPos[2] - this.facingDirection * ((dPos[2] - currPos[2]) * .1),
        );
        // if it is close to character, generate a new position
        if ((this.facingDirection === 1 && dPos[0] < currPos[0] + 1.0)
             || (this.facingDirection === -1 && dPos[0] > currPos[0] - 1.0)) {
          dust.SetPosition(
              currPos[0] + this.facingDirection * (1.5 + Math.random() * 3.0),
              currPos[1] + Math.random() * 2.5,
              currPos[2] + Math.random() * 2.5 + inhalingPosShift,
          );
        }
        dust.RotateX(timeElapsed);
        dust.RotateY(timeElapsed);
      }
    }
  }
}

let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
  _APP = new LightPrepassDemo();
});
