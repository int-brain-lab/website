mergeInto(LibraryManager.library, {

  SelectPID: function (pid) {
    selectPID(UTF8ToString(pid));
  },

  SelectCluster: function (cluster) {
    selectCluster(cluster);
  },

});