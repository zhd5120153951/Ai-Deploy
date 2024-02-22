# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import general_pb2 as general__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


class interactionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.dispatch = channel.unary_unary(
                '/General.interaction/dispatch',
                request_serializer=general__pb2.rsMessage.SerializeToString,
                response_deserializer=general__pb2.rsMessage.FromString,
                )
        self.ping = channel.unary_unary(
                '/General.interaction/ping',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class interactionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def dispatch(self, request, context):
        """调用此函数来传输数据
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ping(self, request, context):
        """测试通信状态
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_interactionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'dispatch': grpc.unary_unary_rpc_method_handler(
                    servicer.dispatch,
                    request_deserializer=general__pb2.rsMessage.FromString,
                    response_serializer=general__pb2.rsMessage.SerializeToString,
            ),
            'ping': grpc.unary_unary_rpc_method_handler(
                    servicer.ping,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'General.interaction', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class interaction(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def dispatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/General.interaction/dispatch',
            general__pb2.rsMessage.SerializeToString,
            general__pb2.rsMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/General.interaction/ping',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
