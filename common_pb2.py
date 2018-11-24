# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto

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
  name='common.proto',
  package='org.beachc.deep_thought.common',
  syntax='proto3',
  serialized_pb=_b('\n\x0c\x63ommon.proto\x12\x1eorg.beachc.deep_thought.common\"S\n\x04Game\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x04path\x18\x02 \x01(\tH\x00\x12\r\n\x03url\x18\x03 \x01(\tH\x00\x12\x12\n\x08rom_data\x18\x04 \x01(\x0cH\x00\x42\n\n\x08location\"\x1d\n\x05Shape\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\".\n\x06Player\x12\x14\n\x0cplayerNumber\x18\x01 \x01(\x05\x12\x0e\n\x06player\x18\x02 \x01(\x05\"=\n\x04\x44Pad\x12\n\n\x02up\x18\x01 \x01(\x08\x12\x0c\n\x04\x64own\x18\x02 \x01(\x08\x12\x0c\n\x04left\x18\x03 \x01(\x08\x12\r\n\x05right\x18\x04 \x01(\x08\x62\x06proto3')
)




_GAME = _descriptor.Descriptor(
  name='Game',
  full_name='org.beachc.deep_thought.common.Game',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='org.beachc.deep_thought.common.Game.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='path', full_name='org.beachc.deep_thought.common.Game.path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='url', full_name='org.beachc.deep_thought.common.Game.url', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rom_data', full_name='org.beachc.deep_thought.common.Game.rom_data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='location', full_name='org.beachc.deep_thought.common.Game.location',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=48,
  serialized_end=131,
)


_SHAPE = _descriptor.Descriptor(
  name='Shape',
  full_name='org.beachc.deep_thought.common.Shape',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='org.beachc.deep_thought.common.Shape.x', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='org.beachc.deep_thought.common.Shape.y', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=133,
  serialized_end=162,
)


_PLAYER = _descriptor.Descriptor(
  name='Player',
  full_name='org.beachc.deep_thought.common.Player',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='playerNumber', full_name='org.beachc.deep_thought.common.Player.playerNumber', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='player', full_name='org.beachc.deep_thought.common.Player.player', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=164,
  serialized_end=210,
)


_DPAD = _descriptor.Descriptor(
  name='DPad',
  full_name='org.beachc.deep_thought.common.DPad',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='up', full_name='org.beachc.deep_thought.common.DPad.up', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='down', full_name='org.beachc.deep_thought.common.DPad.down', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='left', full_name='org.beachc.deep_thought.common.DPad.left', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='right', full_name='org.beachc.deep_thought.common.DPad.right', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=212,
  serialized_end=273,
)

_GAME.oneofs_by_name['location'].fields.append(
  _GAME.fields_by_name['path'])
_GAME.fields_by_name['path'].containing_oneof = _GAME.oneofs_by_name['location']
_GAME.oneofs_by_name['location'].fields.append(
  _GAME.fields_by_name['url'])
_GAME.fields_by_name['url'].containing_oneof = _GAME.oneofs_by_name['location']
_GAME.oneofs_by_name['location'].fields.append(
  _GAME.fields_by_name['rom_data'])
_GAME.fields_by_name['rom_data'].containing_oneof = _GAME.oneofs_by_name['location']
DESCRIPTOR.message_types_by_name['Game'] = _GAME
DESCRIPTOR.message_types_by_name['Shape'] = _SHAPE
DESCRIPTOR.message_types_by_name['Player'] = _PLAYER
DESCRIPTOR.message_types_by_name['DPad'] = _DPAD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Game = _reflection.GeneratedProtocolMessageType('Game', (_message.Message,), dict(
  DESCRIPTOR = _GAME,
  __module__ = 'common_pb2'
  # @@protoc_insertion_point(class_scope:org.beachc.deep_thought.common.Game)
  ))
_sym_db.RegisterMessage(Game)

Shape = _reflection.GeneratedProtocolMessageType('Shape', (_message.Message,), dict(
  DESCRIPTOR = _SHAPE,
  __module__ = 'common_pb2'
  # @@protoc_insertion_point(class_scope:org.beachc.deep_thought.common.Shape)
  ))
_sym_db.RegisterMessage(Shape)

Player = _reflection.GeneratedProtocolMessageType('Player', (_message.Message,), dict(
  DESCRIPTOR = _PLAYER,
  __module__ = 'common_pb2'
  # @@protoc_insertion_point(class_scope:org.beachc.deep_thought.common.Player)
  ))
_sym_db.RegisterMessage(Player)

DPad = _reflection.GeneratedProtocolMessageType('DPad', (_message.Message,), dict(
  DESCRIPTOR = _DPAD,
  __module__ = 'common_pb2'
  # @@protoc_insertion_point(class_scope:org.beachc.deep_thought.common.DPad)
  ))
_sym_db.RegisterMessage(DPad)


# @@protoc_insertion_point(module_scope)
