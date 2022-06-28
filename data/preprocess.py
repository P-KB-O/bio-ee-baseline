from bigbio.dataloader import BigBioConfigHelpers


conhelps = BigBioConfigHelpers()

print("found {} dataset configs from {} datasets".format(
    len(conhelps),
    len(conhelps.available_dataset_names)
))

# list datasets you can get
# print(conhelps.available_dataset_names)

# all dataset config names
ds_config_names = [helper.config.name for helper in conhelps]
print(ds_config_names)

# Loading datasets by config name
mlee_source = conhelps.for_config_name("mlee_source").load_dataset()
mlee_bigbio = conhelps.for_config_name("mlee_bigbio_kb").load_dataset()

print(mlee_source.items())
#
# print(mlee_helper.get_metadata())
