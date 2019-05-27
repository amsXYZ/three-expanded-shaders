#ifdef USE_FOG
	varying float fogDepth;
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHTS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHTS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHTS ];
	#endif
	#if NUM_SPOT_LIGHTS > 0
		uniform mat4 spotShadowMatrix[ NUM_SPOT_LIGHTS ];
		varying vec4 vSpotShadowCoord[ NUM_SPOT_LIGHTS ];
	#endif
	#if NUM_POINT_LIGHTS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHTS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHTS ];
	#endif
#endif
void main() {
	vec3 transformed = vec3( position );
	vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );
gl_Position = projectionMatrix * mvPosition;
	#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP )
	vec4 worldPosition = modelMatrix * vec4( transformed, 1.0 );
#endif
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHTS > 0
	#pragma unroll_loop
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * worldPosition;
	}
	#endif
	#if NUM_SPOT_LIGHTS > 0
	#pragma unroll_loop
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		vSpotShadowCoord[ i ] = spotShadowMatrix[ i ] * worldPosition;
	}
	#endif
	#if NUM_POINT_LIGHTS > 0
	#pragma unroll_loop
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * worldPosition;
	}
	#endif
#endif
	#ifdef USE_FOG
	fogDepth = -mvPosition.z;
#endif
}