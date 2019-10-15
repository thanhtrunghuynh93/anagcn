python -m algorithms.REGAL.REGAL \
--prefix1 ../dataspace/graph/arenas/arenas1/graphsage/arenas \
--prefix2 ../dataspace/graph/arenas/arenas2/graphsage/arenas \
--groundtruth ../dataspace/graph/arenas/dictionaries/groundtruth

python -m algorithms.REGAL.REGAL \
--prefix1 ../dataspace/graph/douban/online/graphsage/online \
--prefix2 ../dataspace/graph/douban/offline/graphsage/offline \
--groundtruth ../dataspace/graph/douban/dictionaries/groundtruth

python -m algorithms.REGAL.REGAL \
--prefix1 ../dataspace/graph/flickr_lastfm/flickr/graphsage/flickr \
--prefix2 ../dataspace/graph/flickr_lastfm/lastfm/graphsage/lastfm \
--groundtruth ../dataspace/graph/flickr_lastfm/dictionaries/groundtruth

python -m algorithms.REGAL.REGAL \
--prefix1 ../dataspace/graph/flickr_myspace/flickr/graphsage/flickr \
--prefix2 ../dataspace/graph/flickr_myspace/myspace/graphsage/myspace \
--groundtruth ../dataspace/graph/flickr_myspace/dictionaries/groundtruth

python -m algorithms.REGAL.REGAL \
--prefix1 ../dataspace/graph/ppi/sub_graph/subgraph0/graphsage/ppi \
--prefix2 ../dataspace/graph/ppi/sub_graph/subgraph0/permut/graphsage/ppi \
--groundtruth ../dataspace/graph/ppi/sub_graph/subgraph0/permut/dictionaries/groundtruth
