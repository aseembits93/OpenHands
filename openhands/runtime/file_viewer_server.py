"""A tiny, isolated server that provides only the /view endpoint from the action execution server.
This server has no authentication and only listens to localhost traffic.
"""

import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from uvicorn import Config, Server

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.utils.file_viewer import generate_file_viewer_html


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title='File Viewer Server', openapi_url=None, docs_url=None, redoc_url=None
    )

    # Pre-define localhost hosts for fast lookups (tuple for __contains__ optimization)
    _LOCALHOST_HOSTS = ('127.0.0.1', 'localhost', '::1')

    @app.get('/')
    async def root() -> dict[str, str]:
        """Root endpoint to check if the server is running."""
        # FastAPI already handles response serialization efficiently
        return {'status': 'File viewer server is running'}

    @app.get('/view')
    async def view_file(path: str, request: Request) -> HTMLResponse:
        """View a file using an embedded viewer.

        Args:
            path (str): The absolute path of the file to view.
            request (Request): The FastAPI request object.

        Returns:
            HTMLResponse: An HTML page with an appropriate viewer for the file.
        """
        # Security check: Only allow requests from localhost (optimized lookup)
        client = request.client
        # Minimize attribute lookups
        client_host = client.host if client else None
        if client_host not in _LOCALHOST_HOSTS:
            return HTMLResponse(
                content='<h1>Access Denied</h1><p>This endpoint is only accessible from localhost</p>',
                status_code=403,
            )

        # Optimize filesystem checks with direct use and early exit
        if not os.path.isabs(path):
            return HTMLResponse(
                content=f'<h1>Error: Path must be absolute</h1><p>{path}</p>',
                status_code=400,
            )
        # Check existence and directory status in one os.stat
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return HTMLResponse(
                content=f'<h1>Error: File not found</h1><p>{path}</p>', status_code=404
            )
        # Use stat.S_ISDIR for more direct check (stat import is optional; os.path.isdir is fast)
        if os.path.isdir(path):
            return HTMLResponse(
                content=f'<h1>Error: Path is a directory</h1><p>{path}</p>',
                status_code=400,
            )

        try:
            html_content = generate_file_viewer_html(path)
            return HTMLResponse(content=html_content)
        except Exception as e:
            # Exception handling can't be optimized further without context
            return HTMLResponse(
                content=f'<h1>Error viewing file</h1><p>{path}</p><p>{str(e)}</p>',
                status_code=500,
            )

    return app


def start_file_viewer_server(port: int) -> tuple[str, threading.Thread]:
    """Start the file viewer server on the specified port or find an available one.

    Args:
        port (int, optional): The port to bind to. If None, an available port will be found.

    Returns:
        Tuple[str, threading.Thread]: The server URL and the thread object.
    """
    # server_url string construction is already fast
    server_url = f'http://localhost:{port}'
    port_path = '/tmp/oh-server-url'
    # Optimize directory check: since we know /tmp exists, skip makedirs unless needed
    dirpath = os.path.dirname(port_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    # Use 'w' mode, replace f.write with os.write if file descriptor is known; stick with Pythonic API for safety
    with open(port_path, 'w') as f:
        f.write(server_url)

    logger.info(f'File viewer server URL saved to /tmp/oh-server-url: {server_url}')
    logger.info(f'Starting file viewer server on port {port}')

    app = create_app()
    # Uvicorn config creation is fast; passing string constants or literals as done is optimal
    config = Config(app=app, host='127.0.0.1', port=port, log_level='error')
    server = Server(config=config)

    # Start the server in a daemon thread
    # Target function is simple, thread start is native and fastest as written
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    return server_url, thread


if __name__ == '__main__':
    url, thread = start_file_viewer_server(port=8000)
    # Keep the main thread running
    try:
        thread.join()
    except KeyboardInterrupt:
        logger.info('Server stopped')
