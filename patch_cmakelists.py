with open('CMakeLists.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Guard sndfile and soxr
t1 = """pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(SOXR REQUIRED IMPORTED_TARGET soxr)"""
r1 = """if (NOT WIN32)
  pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
  pkg_check_modules(SOXR REQUIRED IMPORTED_TARGET soxr)
endif()"""

# 2. Add ornith122b
t2 = "add_subdirectory(src/models/qwen38_flash_next)"
r2 = """add_subdirectory(src/models/qwen38_flash_next)

# Ornith 1.5 122B (qwen35moe) model package.
add_subdirectory(src/models/ornith122b)"""

# 3. Guard embedding_gemma2
t3 = """# EmbeddingGemma 2 is a model-private BF16 text/image embedding runtime.
add_subdirectory(src/models/embedding_gemma2)"""
r3 = """if (NOT WIN32)
  # EmbeddingGemma 2 is a model-private BF16 text/image embedding runtime.
  add_subdirectory(src/models/embedding_gemma2)
endif()"""

t3_link1 = "target_link_libraries(gufo_http PUBLIC gufo_logging gufo_core gufo_qwen3_tts gufo_qwen3_asr gufo_qwen_image_21 gufo_embedding_gemma2)"
r3_link1 = """target_link_libraries(gufo_http PUBLIC gufo_logging gufo_core gufo_qwen3_tts gufo_qwen3_asr gufo_qwen_image_21)
if (NOT WIN32)
  target_link_libraries(gufo_http PUBLIC gufo_embedding_gemma2)
endif()"""

t3_link2 = "target_link_libraries(gufo PRIVATE gufo_qwen3_tts gufo_qwen3_asr gufo_embedding_gemma2)"
r3_link2 = """target_link_libraries(gufo PRIVATE gufo_qwen3_tts gufo_qwen3_asr)
if (NOT WIN32)
  target_link_libraries(gufo PRIVATE gufo_embedding_gemma2)
endif()"""

t3_tests = "add_subdirectory(tests/models/embedding_gemma2)"
r3_tests = """if (NOT WIN32)
  add_subdirectory(tests/models/embedding_gemma2)
endif()"""

# 4. Add ornith122b tests
t4 = "add_subdirectory(tests/models/qwen38_flash_next)"
r4 = """add_subdirectory(tests/models/qwen38_flash_next)
  add_subdirectory(tests/models/ornith122b)"""

assert t1 in content, "t1 not found"
assert t2 in content, "t2 not found"
assert t3 in content, "t3 not found"
assert t3_link1 in content, "t3_link1 not found"
assert t3_link2 in content, "t3_link2 not found"
assert t3_tests in content, "t3_tests not found"
assert t4 in content, "t4 not found"

content = content.replace(t1, r1).replace(t2, r2).replace(t3, r3).replace(t3_link1, r3_link1).replace(t3_link2, r3_link2).replace(t3_tests, r3_tests).replace(t4, r4)

with open('CMakeLists.txt', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
print("Successfully patched CMakeLists.txt")
