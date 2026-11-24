# MCP raster images

This is on by default when the loaded model can read pictures, and off when it cannot. To turn it off, add `"mcp": {"image_results": false}` to the run configuration (beside the other MCP settings); `true` turns it on. The value must be a JSON boolean (`true` or `false`); strings and numbers are rejected. When it is off, the model gets the tool's text only, and the stream events are the same as before.

MCP image results are cached by SHA-256 next to the run configuration, in `<config-stem>.tool-media/`. A client that asked for MCP tools (`"strata_mcp": true`) receives small same-origin references instead of base64 in the conversation. PNG, JPEG and WebP are supported; SVG and arbitrary file/resource URIs are not rendered. Each call displays at most four images, each with at most 24 million base64 characters (about 18 MB decoded). If the cache cannot be written, the chat retains the text result and logs the cache error.

`GET /tool-media/<sha256>.<extension>` uses the existing server API authentication, validates the basename, and serves only cached raster files. A client fetches the images with its usual authorization header. The cache persists across server restarts; if its files are removed, older image references are replaced with an image-omitted text note when preparing later turns. Disabling the feature or removing MCP configuration does the same, while preserving the tool text and allowing the conversation to continue. Malformed references are still rejected. A global disk-retention policy is not implemented yet.

When the server has a vision encoder, tool images enter its existing image preparation path. Later turns send `strata-tool-media:<basename>` references; the authenticated server resolves these only inside its cache. With no vision encoder, the model receives the existing text-only fallback. This adds image transport, not a vision model.

Search and image generation are supplied by configured MCP servers. PolyStrata does not bundle a paid search provider or a ComfyUI installation. Machine-specific adapters and lifecycle hooks are not included in this change.

## Validation without a GPU

```powershell
python -m unittest serve.test_mcp serve.test_mcp_media
git diff --check
```

The transport tests use an existing mock engine and scripted tool response. They verify authenticated binary delivery, invalid media rejection, path traversal refusal, streaming metadata, forwarding to vision preparation and subsequent-turn cache resolution. They do not establish visual understanding or generation quality.
