# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: MWNN.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='MWNN.proto',
  package='MWNN',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n\nMWNN.proto\x12\x04MWNN\">\n\x12MWNNValueInfoProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\x05\x12\x0c\n\x04\x64ims\x18\x03 \x03(\x05\"\xdc\x03\n\x0fMWNNTensorProto\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x05\x12\x11\n\tdata_type\x18\x02 \x01(\x05\x12.\n\x07segment\x18\x03 \x01(\x0b\x32\x1d.MWNN.MWNNTensorProto.Segment\x12\x16\n\nfloat_data\x18\x04 \x03(\x02\x42\x02\x10\x01\x12\x16\n\nint32_data\x18\x05 \x03(\x05\x42\x02\x10\x01\x12\x13\n\x0bstring_data\x18\x06 \x03(\x0c\x12\x16\n\nint64_data\x18\x07 \x03(\x03\x42\x02\x10\x01\x12\x0c\n\x04name\x18\x08 \x01(\t\x12\x17\n\x0buint64_data\x18\x0b \x03(\x04\x42\x02\x10\x01\x1a%\n\x07Segment\x12\r\n\x05\x62\x65gin\x18\x01 \x01(\x03\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x03\"\xcc\x01\n\x08\x44\x61taType\x12\r\n\tUNDEFINED\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\t\n\x05UINT8\x10\x02\x12\x08\n\x04INT8\x10\x03\x12\n\n\x06UINT16\x10\x04\x12\t\n\x05INT16\x10\x05\x12\t\n\x05INT32\x10\x06\x12\t\n\x05INT64\x10\x07\x12\n\n\x06STRING\x10\x08\x12\x08\n\x04\x42OOL\x10\t\x12\x0b\n\x07\x46LOAT16\x10\n\x12\n\n\x06\x44OUBLE\x10\x0b\x12\n\n\x06UINT32\x10\x0c\x12\n\n\x06UINT64\x10\r\x12\r\n\tCOMPLEX64\x10\x0e\x12\x0e\n\nCOMPLEX128\x10\x0f\"L\n\x12MWNNAttributeProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x14 \x01(\x05\x12\x0c\n\x04ints\x18\x08 \x03(\x05\x12\x0c\n\x04\x64\x61ta\x18\t \x03(\t\"\x8a\x01\n\rMWNNNodeProto\x12\r\n\x05input\x18\x01 \x03(\t\x12\x0e\n\x06output\x18\x02 \x03(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x0f\n\x07op_type\x18\x04 \x01(\t\x12\x0e\n\x06\x64omain\x18\x07 \x01(\t\x12+\n\tattribute\x18\x05 \x03(\x0b\x32\x18.MWNN.MWNNAttributeProto\"\xeb\x01\n\x0eMWNNGraphProto\x12!\n\x04node\x18\x01 \x03(\x0b\x32\x13.MWNN.MWNNNodeProto\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x13\n\x0bgraph_input\x18\x03 \x01(\t\x12\x14\n\x0cgraph_output\x18\x04 \x01(\t\x12*\n\x0binitializer\x18\x05 \x03(\x0b\x32\x15.MWNN.MWNNTensorProto\x12\'\n\x05input\x18\x0b \x03(\x0b\x32\x18.MWNN.MWNNValueInfoProto\x12(\n\x06output\x18\x0c \x03(\x0b\x32\x18.MWNN.MWNNValueInfoProto'
)



_MWNNTENSORPROTO_DATATYPE = _descriptor.EnumDescriptor(
  name='DataType',
  full_name='MWNN.MWNNTensorProto.DataType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT8', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT8', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT16', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT16', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT32', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT64', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STRING', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BOOL', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT16', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOUBLE', index=11, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT32', index=12, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT64', index=13, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COMPLEX64', index=14, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COMPLEX128', index=15, number=15,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=357,
  serialized_end=561,
)
_sym_db.RegisterEnumDescriptor(_MWNNTENSORPROTO_DATATYPE)


_MWNNVALUEINFOPROTO = _descriptor.Descriptor(
  name='MWNNValueInfoProto',
  full_name='MWNN.MWNNValueInfoProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='MWNN.MWNNValueInfoProto.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='MWNN.MWNNValueInfoProto.type', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dims', full_name='MWNN.MWNNValueInfoProto.dims', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20,
  serialized_end=82,
)


_MWNNTENSORPROTO_SEGMENT = _descriptor.Descriptor(
  name='Segment',
  full_name='MWNN.MWNNTensorProto.Segment',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='begin', full_name='MWNN.MWNNTensorProto.Segment.begin', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='MWNN.MWNNTensorProto.Segment.end', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=317,
  serialized_end=354,
)

_MWNNTENSORPROTO = _descriptor.Descriptor(
  name='MWNNTensorProto',
  full_name='MWNN.MWNNTensorProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dims', full_name='MWNN.MWNNTensorProto.dims', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_type', full_name='MWNN.MWNNTensorProto.data_type', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='segment', full_name='MWNN.MWNNTensorProto.segment', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_data', full_name='MWNN.MWNNTensorProto.float_data', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int32_data', full_name='MWNN.MWNNTensorProto.int32_data', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='string_data', full_name='MWNN.MWNNTensorProto.string_data', index=5,
      number=6, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64_data', full_name='MWNN.MWNNTensorProto.int64_data', index=6,
      number=7, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='MWNN.MWNNTensorProto.name', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='uint64_data', full_name='MWNN.MWNNTensorProto.uint64_data', index=8,
      number=11, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MWNNTENSORPROTO_SEGMENT, ],
  enum_types=[
    _MWNNTENSORPROTO_DATATYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=561,
)


_MWNNATTRIBUTEPROTO = _descriptor.Descriptor(
  name='MWNNAttributeProto',
  full_name='MWNN.MWNNAttributeProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='MWNN.MWNNAttributeProto.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='MWNN.MWNNAttributeProto.type', index=1,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ints', full_name='MWNN.MWNNAttributeProto.ints', index=2,
      number=8, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='MWNN.MWNNAttributeProto.data', index=3,
      number=9, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=563,
  serialized_end=639,
)


_MWNNNODEPROTO = _descriptor.Descriptor(
  name='MWNNNodeProto',
  full_name='MWNN.MWNNNodeProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input', full_name='MWNN.MWNNNodeProto.input', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output', full_name='MWNN.MWNNNodeProto.output', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='MWNN.MWNNNodeProto.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op_type', full_name='MWNN.MWNNNodeProto.op_type', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='domain', full_name='MWNN.MWNNNodeProto.domain', index=4,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attribute', full_name='MWNN.MWNNNodeProto.attribute', index=5,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=642,
  serialized_end=780,
)


_MWNNGRAPHPROTO = _descriptor.Descriptor(
  name='MWNNGraphProto',
  full_name='MWNN.MWNNGraphProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='node', full_name='MWNN.MWNNGraphProto.node', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='MWNN.MWNNGraphProto.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='graph_input', full_name='MWNN.MWNNGraphProto.graph_input', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='graph_output', full_name='MWNN.MWNNGraphProto.graph_output', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initializer', full_name='MWNN.MWNNGraphProto.initializer', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input', full_name='MWNN.MWNNGraphProto.input', index=5,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output', full_name='MWNN.MWNNGraphProto.output', index=6,
      number=12, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=783,
  serialized_end=1018,
)

_MWNNTENSORPROTO_SEGMENT.containing_type = _MWNNTENSORPROTO
_MWNNTENSORPROTO.fields_by_name['segment'].message_type = _MWNNTENSORPROTO_SEGMENT
_MWNNTENSORPROTO_DATATYPE.containing_type = _MWNNTENSORPROTO
_MWNNNODEPROTO.fields_by_name['attribute'].message_type = _MWNNATTRIBUTEPROTO
_MWNNGRAPHPROTO.fields_by_name['node'].message_type = _MWNNNODEPROTO
_MWNNGRAPHPROTO.fields_by_name['initializer'].message_type = _MWNNTENSORPROTO
_MWNNGRAPHPROTO.fields_by_name['input'].message_type = _MWNNVALUEINFOPROTO
_MWNNGRAPHPROTO.fields_by_name['output'].message_type = _MWNNVALUEINFOPROTO
DESCRIPTOR.message_types_by_name['MWNNValueInfoProto'] = _MWNNVALUEINFOPROTO
DESCRIPTOR.message_types_by_name['MWNNTensorProto'] = _MWNNTENSORPROTO
DESCRIPTOR.message_types_by_name['MWNNAttributeProto'] = _MWNNATTRIBUTEPROTO
DESCRIPTOR.message_types_by_name['MWNNNodeProto'] = _MWNNNODEPROTO
DESCRIPTOR.message_types_by_name['MWNNGraphProto'] = _MWNNGRAPHPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MWNNValueInfoProto = _reflection.GeneratedProtocolMessageType('MWNNValueInfoProto', (_message.Message,), {
  'DESCRIPTOR' : _MWNNVALUEINFOPROTO,
  '__module__' : 'MWNN_pb2'
  # @@protoc_insertion_point(class_scope:MWNN.MWNNValueInfoProto)
  })
_sym_db.RegisterMessage(MWNNValueInfoProto)

MWNNTensorProto = _reflection.GeneratedProtocolMessageType('MWNNTensorProto', (_message.Message,), {

  'Segment' : _reflection.GeneratedProtocolMessageType('Segment', (_message.Message,), {
    'DESCRIPTOR' : _MWNNTENSORPROTO_SEGMENT,
    '__module__' : 'MWNN_pb2'
    # @@protoc_insertion_point(class_scope:MWNN.MWNNTensorProto.Segment)
    })
  ,
  'DESCRIPTOR' : _MWNNTENSORPROTO,
  '__module__' : 'MWNN_pb2'
  # @@protoc_insertion_point(class_scope:MWNN.MWNNTensorProto)
  })
_sym_db.RegisterMessage(MWNNTensorProto)
_sym_db.RegisterMessage(MWNNTensorProto.Segment)

MWNNAttributeProto = _reflection.GeneratedProtocolMessageType('MWNNAttributeProto', (_message.Message,), {
  'DESCRIPTOR' : _MWNNATTRIBUTEPROTO,
  '__module__' : 'MWNN_pb2'
  # @@protoc_insertion_point(class_scope:MWNN.MWNNAttributeProto)
  })
_sym_db.RegisterMessage(MWNNAttributeProto)

MWNNNodeProto = _reflection.GeneratedProtocolMessageType('MWNNNodeProto', (_message.Message,), {
  'DESCRIPTOR' : _MWNNNODEPROTO,
  '__module__' : 'MWNN_pb2'
  # @@protoc_insertion_point(class_scope:MWNN.MWNNNodeProto)
  })
_sym_db.RegisterMessage(MWNNNodeProto)

MWNNGraphProto = _reflection.GeneratedProtocolMessageType('MWNNGraphProto', (_message.Message,), {
  'DESCRIPTOR' : _MWNNGRAPHPROTO,
  '__module__' : 'MWNN_pb2'
  # @@protoc_insertion_point(class_scope:MWNN.MWNNGraphProto)
  })
_sym_db.RegisterMessage(MWNNGraphProto)


_MWNNTENSORPROTO.fields_by_name['float_data']._options = None
_MWNNTENSORPROTO.fields_by_name['int32_data']._options = None
_MWNNTENSORPROTO.fields_by_name['int64_data']._options = None
_MWNNTENSORPROTO.fields_by_name['uint64_data']._options = None
# @@protoc_insertion_point(module_scope)