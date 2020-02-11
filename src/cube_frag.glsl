#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform int maxMipLevel;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif
uniform float opacity;
varying vec3 vWorldDirection;
#ifdef ENVMAP_TYPE_CUBE_UV
#define cubeUV_maxMipLevel 8.0
#define cubeUV_minMipLevel 4.0
#define cubeUV_maxTileSize 256.0
#define cubeUV_minTileSize 16.0
float getFace(vec3 direction) {
    vec3 absDirection = abs(direction);
    float face = -1.0;
    if (absDirection.x > absDirection.z) {
      if (absDirection.x > absDirection.y)
        face = direction.x > 0.0 ? 0.0 : 3.0;
      else
        face = direction.y > 0.0 ? 1.0 : 4.0;
    } else {
      if (absDirection.z > absDirection.y)
        face = direction.z > 0.0 ? 2.0 : 5.0;
      else
        face = direction.y > 0.0 ? 1.0 : 4.0;
    }
    return face;
}
vec2 getUV(vec3 direction, float face) {
    vec2 uv;
    if (face == 0.0) {
      uv = vec2(-direction.z, direction.y) / abs(direction.x);
    } else if (face == 1.0) {
      uv = vec2(direction.x, -direction.z) / abs(direction.y);
    } else if (face == 2.0) {
      uv = direction.xy / abs(direction.z);
    } else if (face == 3.0) {
      uv = vec2(direction.z, direction.y) / abs(direction.x);
    } else if (face == 4.0) {
      uv = direction.xz / abs(direction.y);
    } else {
      uv = vec2(-direction.x, direction.y) / abs(direction.z);
    }
    return 0.5 * (uv + 1.0);
}
vec3 bilinearCubeUV(sampler2D envMap, vec3 direction, float mipInt) {
  float face = getFace(direction);
  float filterInt = max(cubeUV_minMipLevel - mipInt, 0.0);
  mipInt = max(mipInt, cubeUV_minMipLevel);
  float faceSize = exp2(mipInt);
  float texelSize = 1.0 / (3.0 * cubeUV_maxTileSize);
  vec2 uv = getUV(direction, face) * (faceSize - 1.0);
  vec2 f = fract(uv);
  uv += 0.5 - f;
  if (face > 2.0) {
    uv.y += faceSize;
    face -= 3.0;
  }
  uv.x += face * faceSize;
  if(mipInt < cubeUV_maxMipLevel){
    uv.y += 2.0 * cubeUV_maxTileSize;
  }
  uv.y += filterInt * 2.0 * cubeUV_minTileSize;
  uv.x += 3.0 * max(0.0, cubeUV_maxTileSize - 2.0 * faceSize);
  uv *= texelSize;
  vec3 tl = envMapTexelToLinear(texture2D(envMap, uv)).rgb;
  uv.x += texelSize;
  vec3 tr = envMapTexelToLinear(texture2D(envMap, uv)).rgb;
  uv.y += texelSize;
  vec3 br = envMapTexelToLinear(texture2D(envMap, uv)).rgb;
  uv.x -= texelSize;
  vec3 bl = envMapTexelToLinear(texture2D(envMap, uv)).rgb;
  vec3 tm = mix(tl, tr, f.x);
  vec3 bm = mix(bl, br, f.x);
  return mix(tm, bm, f.y);
}
#define r0 1.0
#define v0 0.339
#define m0 -2.0
#define r1 0.8
#define v1 0.276
#define m1 -1.0
#define r4 0.4
#define v4 0.046
#define m4 2.0
#define r5 0.305
#define v5 0.016
#define m5 3.0
#define r6 0.21
#define v6 0.0038
#define m6 4.0
float roughnessToMip(float roughness) {
  float mip = 0.0;
  if (roughness >= r1) {
    mip = (r0 - roughness) * (m1 - m0) / (r0 - r1) + m0;
  } else if (roughness >= r4) {
    mip = (r1 - roughness) * (m4 - m1) / (r1 - r4) + m1;
  } else if (roughness >= r5) {
    mip = (r4 - roughness) * (m5 - m4) / (r4 - r5) + m4;
  } else if (roughness >= r6) {
    mip = (r5 - roughness) * (m6 - m5) / (r5 - r6) + m5;
  } else {
    mip = -2.0 * log2(1.16 * roughness);  }
  return mip;
}
vec4 textureCubeUV(sampler2D envMap, vec3 sampleDir, float roughness) {
  float mip = clamp(roughnessToMip(roughness), m0, cubeUV_maxMipLevel);
  float mipF = fract(mip);
  float mipInt = floor(mip);
  vec3 color0 = bilinearCubeUV(envMap, sampleDir, mipInt);
  if (mipF == 0.0) {
    return vec4(color0, 1.0);
  } else {
    vec3 color1 = bilinearCubeUV(envMap, sampleDir, mipInt + 1.0);
    return vec4(mix(color0, color1, mipF), 1.0);
  }
}
#endif
void main() {
	vec3 vReflect = vWorldDirection;
	#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		}  else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 envColor = textureCubeUV( envMap, vec3( flipEnvMap * reflectVec.x, reflectVec.yz ), 0.0 );
	#elif defined( ENVMAP_TYPE_EQUIREC )
		vec2 sampleUV;
		reflectVec = normalize( reflectVec );
		sampleUV.y = asin( clamp( reflectVec.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
		sampleUV.x = atan( reflectVec.z, reflectVec.x ) * RECIPROCAL_PI2 + 0.5;
		vec4 envColor = texture2D( envMap, sampleUV );
	#elif defined( ENVMAP_TYPE_SPHERE )
		reflectVec = normalize( reflectVec );
		vec3 reflectView = normalize( ( viewMatrix * vec4( reflectVec, 0.0 ) ).xyz + vec3( 0.0, 0.0, 1.0 ) );
		vec4 envColor = texture2D( envMap, reflectView.xy * 0.5 + 0.5 );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifndef ENVMAP_TYPE_CUBE_UV
		envColor = envMapTexelToLinear( envColor );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif
	gl_FragColor = envColor;
	gl_FragColor.a *= opacity;
	#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif
	gl_FragColor = linearToOutputTexel( gl_FragColor );
}