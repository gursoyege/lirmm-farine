from common.imports import *

def clean():
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    if os.path.exists(TEST_OUT_PATH):
        shutil.rmtree(TEST_OUT_PATH)
def run():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(TEST_OUT_PATH):
        os.makedirs(TEST_OUT_PATH)

