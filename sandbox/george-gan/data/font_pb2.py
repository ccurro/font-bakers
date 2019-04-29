# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: font.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='font.proto',
  package='',
  serialized_pb=_b('\n\nfont.proto\"\x1d\n\x05glyph\x12\x14\n\x05glyph\x18\x01 \x03(\x0b\x32\x05.Font\"u\n\x04\x46ont\x12\x14\n\x0cnum_contours\x18\x01 \x01(\x05\x12\x15\n\rbezier_points\x18\x02 \x03(\x02\x12\x19\n\x11\x63ontour_locations\x18\x03 \x03(\x05\x12\x11\n\tfont_name\x18\x04 \x01(\t\x12\x12\n\nglyph_name\x18\x05 \x01(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_GLYPH = _descriptor.Descriptor(
  name='glyph',
  full_name='glyph',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='glyph', full_name='glyph.glyph', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=14,
  serialized_end=43,
)


_FONT = _descriptor.Descriptor(
  name='Font',
  full_name='Font',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_contours', full_name='Font.num_contours', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bezier_points', full_name='Font.bezier_points', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='contour_locations', full_name='Font.contour_locations', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='font_name', full_name='Font.font_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='glyph_name', full_name='Font.glyph_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=162,
)

_GLYPH.fields_by_name['glyph'].message_type = _FONT
DESCRIPTOR.message_types_by_name['glyph'] = _GLYPH
DESCRIPTOR.message_types_by_name['Font'] = _FONT

glyph = _reflection.GeneratedProtocolMessageType('glyph', (_message.Message,), dict(
  DESCRIPTOR = _GLYPH,
  __module__ = 'font_pb2'
  # @@protoc_insertion_point(class_scope:glyph)
  ))
_sym_db.RegisterMessage(glyph)

Font = _reflection.GeneratedProtocolMessageType('Font', (_message.Message,), dict(
  DESCRIPTOR = _FONT,
  __module__ = 'font_pb2'
  # @@protoc_insertion_point(class_scope:Font)
  ))
_sym_db.RegisterMessage(Font)


# @@protoc_insertion_point(module_scope)
