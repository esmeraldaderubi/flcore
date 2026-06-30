"""
Monkey-patches for Flower/gRPC compatibility with Traefik-based deployments.

These patches are required for multi-client routing through Traefik and must be
applied before Flower opens any gRPC connections. They are isolated here to
make their scope and intent explicit, and to keep server.py and client.py free
of low-level networking internals.

Flower version compatibility: tested with Flower 1.x (1.10–1.22).
If a Flower upgrade breaks these patches:
  - For patch_server_cid: update CANDIDATE_PATHS to match the new module layout.
  - For patch_flwr_channel: update _FLWR_CONNECTION_MODULES.
  Run inside the container to discover current paths:
    python -c "
    import flwr, sys
    [print(n, a) for n, m in sys.modules.items()
     if n.startswith('flwr')
     for a in dir(m)
     if 'Servicer' in a and hasattr(getattr(m, a, None), 'Join')]"
"""

import importlib
import logging
import sys

import grpc

_log = logging.getLogger(__name__)


# ==============================================================================
# CLIENT-SIDE: gRPC execution-id interceptor  *** DO NOT REMOVE ***
#
# WHY: Injects the `execution-id` header into every gRPC call so Traefik can
# route traffic to the correct Flower server container on GROOT. Also injects
# `client-id` so the server-side CID patch can assign unique identities to
# clients that arrive behind a shared Traefik upstream.
# ==============================================================================

class _ClientCallDetails(grpc.ClientCallDetails):
    """Writable wrapper around grpc.ClientCallDetails."""

    def __init__(self, method, timeout, metadata, credentials):
        self.method = method
        self.timeout = timeout
        self.metadata = metadata
        self.credentials = credentials


class ExecutionIdInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """
    Injects `execution-id` and `client-id` gRPC metadata headers into every
    outgoing call. Traefik reads `execution-id` to route to the correct Flower
    server container; the server reads `client-id` to assign a stable CID.
    """

    def __init__(self, execution_id: str, client_id: str):
        self._metadata = [("execution-id", execution_id), ("client-id", client_id)]

    def _augment(self, client_call_details: grpc.ClientCallDetails):
        metadata = list(client_call_details.metadata or [])
        metadata = [m for m in metadata if m[0] not in ("execution-id", "client-id")]
        metadata.extend(self._metadata)
        return _ClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=metadata,
            credentials=client_call_details.credentials,
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._augment(client_call_details), request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return continuation(self._augment(client_call_details), request_iterator)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._augment(client_call_details), request)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        metadata = list(client_call_details.metadata or [])
        metadata = [m for m in metadata if m[0] not in ("execution-id", "client-id")]
        metadata.extend(self._metadata)
        new_details = _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials,
        )
        return continuation(new_details, request_iterator)


# Flower modules that cache a reference to create_channel at import time.
# Pre-importing them ensures our patch below covers all of them.
_FLWR_CONNECTION_MODULES = [
    "flwr.compat.client.grpc_client.connection",   # 1.22.x
    "flwr.client.grpc_client.connection",           # older 1.x
    "flwr.cli.utils",
    "flwr.server.grid.grpc_grid",
    "flwr.simulation.simulationio_connection",
    "flwr.supercore.superexec.run_superexec",
]


def _preload_flwr_modules() -> None:
    """Pre-import Flower modules so our create_channel patch covers all of them."""
    for mod_path in _FLWR_CONNECTION_MODULES:
        try:
            importlib.import_module(mod_path)
        except ImportError:
            pass


def patch_flwr_channel(
    execution_id: str,
    client_id: str,
    root_certificate: bytes = None,
) -> None:
    """
    Monkey-patch Flower's create_channel to inject our interceptor and handle
    version-specific arguments like 'max_message_length'.

    Must be called BEFORE fl.client.start_numpy_client / start_client.

    - If root_certificate is provided: TLS secure channel (client → Traefik).
    - Otherwise: insecure channel (local/dev mode only).
    The Traefik → Flower-server leg is always h2c (cleartext) since both
    run on the same host; only the external client-to-Traefik leg needs TLS.
    """
    import flwr.common.grpc as _flwr_grpc

    _preload_flwr_modules()

    def patched_create_channel(server_address, root_certificates=None, options=None, **kwargs):
        options = list(options) if options else []
        max_msg_len = kwargs.get("max_message_length") or kwargs.get("grpc_max_message_length")
        if max_msg_len:
            options.append(("grpc.max_receive_message_length", max_msg_len))
            options.append(("grpc.max_send_message_length", max_msg_len))

        if root_certificate:
            _log.info(f"[Client] Creating TLS channel to {server_address} with options: {options}")
            credentials = grpc.ssl_channel_credentials(root_certificates=root_certificate)
            channel = grpc.secure_channel(server_address, credentials, options=options)
        else:
            _log.info(f"[Client] Creating insecure channel to {server_address} with options: {options}")
            channel = grpc.insecure_channel(
                server_address,
                options=options,
                compression=kwargs.get("compression"),
            )

        interceptor = ExecutionIdInterceptor(execution_id, client_id)
        return grpc.intercept_channel(channel, interceptor)

    _flwr_grpc.create_channel = patched_create_channel

    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("flwr") and mod is not _flwr_grpc:
            if hasattr(mod, "create_channel"):
                setattr(mod, "create_channel", patched_create_channel)
                _log.info(f"[Client] Patched create_channel in {mod_name}")


# ==============================================================================
# SERVER-SIDE: CID patch for Traefik multi-client routing  *** DO NOT REMOVE ***
#
# WHY: When clients connect via Traefik, all streams share the same upstream
# TCP connection regardless of separate Traefik services. context.peer() returns
# Traefik's IP for every client, so SimpleClientManager sees them all as the
# same client and immediately closes duplicate Join streams (StopIteration).
# This patch overrides peer() per-stream with the 'client-id' gRPC header
# injected by the client's ExecutionIdInterceptor, giving each client a unique
# stable CID regardless of transport topology.
# ==============================================================================

def patch_server_cid(logger) -> None:
    """
    Monkey-patch Flower's gRPC Join handler to use the 'client-id' gRPC metadata
    header as the CID instead of context.peer().

    Tries all known module locations across Flower 1.x versions, because the
    class moves between releases and a hard import path silently skips when
    the version doesn't match.
    """
    CANDIDATE_PATHS = [
        ("flwr.server.grpc_server.flower_service_servicer", "FlowerServiceServicer"),
        ("flwr.server.grpc_server",                         "FlowerServiceServicer"),
        ("flwr.server.fleet.grpc_bidi.flower_service_servicer", "FlowerServiceServicer"),
        ("flwr.server.fleet.grpc_bidi",                     "FlowerServiceServicer"),
        ("flwr.compat.server.grpc_server.flower_service_servicer", "FlowerServiceServicer"),
        ("flwr.compat.server.grpc_server",                  "FlowerServiceServicer"),
    ]

    target_cls = None
    found_at = None

    for mod_path, cls_name in CANDIDATE_PATHS:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "Join"):
                target_cls = cls
                found_at = f"{mod_path}.{cls_name}"
                break
        except ImportError:
            continue

    if target_cls is None:
        for mod_name, mod in list(sys.modules.items()):
            if not mod_name.startswith("flwr"):
                continue
            for attr_name in dir(mod):
                cls = getattr(mod, attr_name, None)
                if isinstance(cls, type) and "Servicer" in attr_name and hasattr(cls, "Join"):
                    target_cls = cls
                    found_at = f"{mod_name}.{attr_name}"
                    break
            if target_cls:
                break

    if target_cls is None:
        logger.error(
            "[Server] CID patch FAILED — could not locate FlowerServiceServicer "
            "in any known module. Multi-client Traefik routing will NOT work.\n"
            "Run this inside the container to find the correct path:\n"
            "  python -c \""
            "import flwr, sys; "
            "[print(n, a) for n, m in sys.modules.items() "
            "if n.startswith('flwr') "
            "for a in dir(m) "
            "if 'Servicer' in a and hasattr(getattr(m,a,None),'Join')]\""
        )
        return

    logger.info(f"[Server] Found FlowerServiceServicer at: {found_at}")
    _original_join = target_cls.Join

    def _patched_join(self, request_iterator, context):
        metadata = dict(context.invocation_metadata())
        client_id_from_header = metadata.get("client-id")
        if client_id_from_header:
            original_peer = context.peer
            context.peer = lambda: f"client-id:{client_id_from_header}"
            logger.info(
                f"[Server] Join: using client-id='{client_id_from_header}' as CID "
                f"(was peer={original_peer()})"
            )
            try:
                yield from _original_join(self, request_iterator, context)
            finally:
                context.peer = original_peer
        else:
            logger.warning(
                f"[Server] Join: no 'client-id' header — "
                f"falling back to peer={context.peer()}"
            )
            yield from _original_join(self, request_iterator, context)

    target_cls.Join = _patched_join
    logger.info(f"[Server] Patched {found_at}.Join to use 'client-id' metadata as CID")
 