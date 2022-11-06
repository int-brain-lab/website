using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.AddressableAssets.Settings.GroupSchemas;
using UnityEditor.AddressableAssets.Settings;
using UnityEditor.AddressableAssets;
using UnityEditor.Graphs;
using UnityEngine;
using System.IO;

public class MenuItems
{
    [MenuItem("Tools/BuildGroups")]
    private static void BuildAddressableGroups()
    {
        Debug.Log("Re-building addressable groups");
        string[] folders = AssetDatabase.GetSubFolders("Assets/AddressableAssets");

        foreach (string folder in folders)
        {
            string pid = folder.Substring(25);
            Debug.Log(string.Format("Found asset folder with pid {0}", pid));
            // Find all the assets
            string[] assetGUIDs = AssetDatabase.FindAssets(pid, new string[] { folder });

            var settings = AddressableAssetSettingsDefaultObject.Settings;

            if (settings)
            {
                var group = settings.FindGroup(pid);
                if (!group)
                    group = settings.CreateGroup(pid, false, false, true, null, typeof(ContentUpdateGroupSchema), typeof(BundledAssetGroupSchema));

                foreach (string guid in assetGUIDs)
                {
                    //Debug.Log(asset);
                    //var assetpath = AssetDatabase.GetAssetPath(obj);
                    //var guid = AssetDatabase.AssetPathToGUID(assetpath);

                    var e = settings.CreateOrMoveEntry(guid, group, false, false);
                    var entriesAdded = new List<AddressableAssetEntry> { e };

                    group.SetDirty(AddressableAssetSettings.ModificationEvent.EntryMoved, entriesAdded, false, true);
                    settings.SetDirty(AddressableAssetSettings.ModificationEvent.EntryMoved, entriesAdded, true, false);
                }
            }
        }

    }
}