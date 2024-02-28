from one.api import ONE
from brainbox.io.one import EphysSessionLoader

one = ONE()

eid = "c8d46ee6-eb68-4535-8756-7c9aa32f10e4"
esl = EphysSessionLoader(eid=eid, one=one)
esl.load_session_data()

# # List datasets associated with a session, in the alf collection
# datasets = one.list_datasets(eid, collection='alf*')

# # Download all data in alf collection
# files = one.load_collection(eid, 'alf', download_only=True)

# # Show where files have been downloaded to
# print(f'Files downloaded to {files[0].parent}')
