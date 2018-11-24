# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: snes.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='snes.proto',
  package='org.beachc.deep_thought.snes',
  syntax='proto3',
  serialized_pb=_b('\n\nsnes.proto\x12\x1corg.beachc.deep_thought.snes\x1a\x0c\x63ommon.proto\"\x8c\x02\n\x10SNESConsoleState\x12H\n\rplayer1_input\x18\x01 \x01(\x0b\x32\x31.org.beachc.deep_thought.snes.SNESControllerState\x12H\n\rplayer2_input\x18\x02 \x01(\x0b\x32\x31.org.beachc.deep_thought.snes.SNESControllerState\x12\r\n\x05power\x18\x03 \x01(\x08\x12\r\n\x05reset\x18\x04 \x01(\x08\x12\x12\n\ninsertGame\x18\x05 \x01(\x08\x12\x32\n\x04game\x18\x06 \x01(\x0b\x32$.org.beachc.deep_thought.common.Game\"\xe2\x01\n\x13SNESControllerState\x12\x32\n\x04\x64pad\x18\x01 \x01(\x0b\x32$.org.beachc.deep_thought.common.DPad\x12\x0e\n\x06select\x18\x02 \x01(\x08\x12\r\n\x05start\x18\x03 \x01(\x08\x12\t\n\x01\x61\x18\x04 \x01(\x08\x12\t\n\x01\x62\x18\x05 \x01(\x08\x12\t\n\x01x\x18\x06 \x01(\x08\x12\t\n\x01y\x18\x07 \x01(\x08\x12\t\n\x01r\x18\x08 \x01(\x08\x12\t\n\x01l\x18\t \x01(\x08\x12\x36\n\x06player\x18\n \x01(\x0b\x32&.org.beachc.deep_thought.common.Playerb\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])




_SNESCONSOLESTATE = _descriptor.Descriptor(
  name='SNESConsoleState',
  full_name='org.beachc.deep_thought.snes.SNESConsoleState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='player1_input', full_name='org.beachc.deep_thought.snes.SNESConsoleState.player1_input', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='player2_input', full_name='org.beachc.deep_thought.snes.SNESConsoleState.player2_input', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power', full_name='org.beachc.deep_thought.snes.SNESConsoleState.power', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reset', full_name='org.beachc.deep_thought.snes.SNESConsoleState.reset', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='insertGame', full_name='org.beachc.deep_thought.snes.SNESConsoleState.insertGame', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='game', full_name='org.beachc.deep_thought.snes.SNESConsoleState.game', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=59,
  serialized_end=327,
)


_SNESCONTROLLERSTATE = _descriptor.Descriptor(
  name='SNESControllerState',
  full_name='org.beachc.deep_thought.snes.SNESControllerState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dpad', full_name='org.beachc.deep_thought.snes.SNESControllerState.dpad', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='select', full_name='org.beachc.deep_thought.snes.SNESControllerState.select', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='org.beachc.deep_thought.snes.SNESControllerState.start', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='a', full_name='org.beachc.deep_thought.snes.SNESControllerState.a', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='org.beachc.deep_thought.snes.SNESControllerState.b', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='x', full_name='org.beachc.deep_thought.snes.SNESControllerState.x', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='org.beachc.deep_thought.snes.SNESControllerState.y', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='r', full_name='org.beachc.deep_thought.snes.SNESControllerState.r', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='l', full_name='org.beachc.deep_thought.snes.SNESControllerState.l', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='player', full_name='org.beachc.deep_thought.snes.SNESControllerState.player', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=330,
  serialized_end=556,
)

_SNESCONSOLESTATE.fields_by_name['player1_input'].message_type = _SNESCONTROLLERSTATE
_SNESCONSOLESTATE.fields_by_name['player2_input'].message_type = _SNESCONTROLLERSTATE
_SNESCONSOLESTATE.fields_by_name['game'].message_type = common__pb2._GAME
_SNESCONTROLLERSTATE.fields_by_name['dpad'].message_type = common__pb2._DPAD
_SNESCONTROLLERSTATE.fields_by_name['player'].message_type = common__pb2._PLAYER
DESCRIPTOR.message_types_by_name['SNESConsoleState'] = _SNESCONSOLESTATE
DESCRIPTOR.message_types_by_name['SNESControllerState'] = _SNESCONTROLLERSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SNESConsoleState = _reflection.GeneratedProtocolMessageType('SNESConsoleState', (_message.Message,), dict(
  DESCRIPTOR = _SNESCONSOLESTATE,
  __module__ = 'snes_pb2'
  # @@protoc_insertion_point(class_scope:org.beachc.deep_thought.snes.SNESConsoleState)
  ))
_sym_db.RegisterMessage(SNESConsoleState)

SNESControllerState = _reflection.GeneratedProtocolMessageType('SNESControllerState', (_message.Message,), dict(
  DESCRIPTOR = _SNESCONTROLLERSTATE,
  __module__ = 'snes_pb2'
  # @@protoc_insertion_point(class_scope:org.beachc.deep_thought.snes.SNESControllerState)
  ))
_sym_db.RegisterMessage(SNESControllerState)


# @@protoc_insertion_point(module_scope)
