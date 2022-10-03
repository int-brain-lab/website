using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.ResourceManagement.AsyncOperations;

public class NPYReader : MonoBehaviour
{
    public static async Task<float[]> LoadBinaryFloatHelper(string path)
    {
        AsyncOperationHandle<TextAsset> dataHandle = Addressables.LoadAssetAsync<TextAsset>(path);
        await dataHandle.Task;

        int nBytes = dataHandle.Result.bytes.Length;
        Debug.Log(string.Format("Loading {0} with {1} bytes", path, nBytes));
        float[] data = new float[nBytes / 4];

        Buffer.BlockCopy(dataHandle.Result.bytes, 0, data, 0, nBytes);
        Debug.LogFormat("Found {0} floats", data.Length);

        return data;
    }
}
