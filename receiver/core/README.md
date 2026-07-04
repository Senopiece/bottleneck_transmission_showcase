# Core

Portable C++ code goes here.

The goal is to keep platform adapters thin:

- Android passes camera frames to this core through JNI.
- Desktop tools pass decoded video frames to this core.
- Future iOS can wrap the same logic with ObjC++/Swift.
