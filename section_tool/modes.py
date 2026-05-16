from enum import Enum, auto


class Mode(Enum):
    SECTION = auto()
    MAP     = auto()
    THREE_D = auto()


MINIMAP_SOURCES = {
    Mode.SECTION: Mode.MAP,
    Mode.MAP:     Mode.SECTION,
    Mode.THREE_D: Mode.SECTION,
}
