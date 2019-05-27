const three = require("three");
const fs = require("fs");

function main() {
    const includeRegExp = new RegExp(/#include[ \t]+["<]([a-zA-Z0-9-_./ ]+)[">]/);
    const shaders = Object.keys(three.ShaderLib);
    for (const shader of shaders) {
        // Vertex Shader Expansion.
        let includeVS = includeRegExp.exec(three.ShaderLib[shader].vertexShader);
        while (includeVS !== null) {
            three.ShaderLib[shader].vertexShader = three.ShaderLib[shader].vertexShader.replace(
                includeRegExp,
                three.ShaderChunk[includeVS[1]]
            );
            includeVS = includeRegExp.exec(three.ShaderLib[shader].vertexShader);
        }
        fs.writeFile(
            `./src/${shader}_vert.glsl`,
            three.ShaderLib[shader].vertexShader,
            "utf8",
            () => {}
        );

        // Fragment Shader Expansion.
        let includeFS = includeRegExp.exec(three.ShaderLib[shader].fragmentShader);
        while (includeFS !== null) {
            three.ShaderLib[shader].fragmentShader = three.ShaderLib[shader].fragmentShader.replace(
                includeRegExp,
                three.ShaderChunk[includeFS[1]]
            );
            includeFS = includeRegExp.exec(three.ShaderLib[shader].fragmentShader);
        }
        fs.writeFile(
            `./src/${shader}_frag.glsl`,
            three.ShaderLib[shader].fragmentShader,
            "utf8",
            () => {}
        );
    }
}
main();
