add_library(myslam SHARED 
                                config.cpp
                                system.cpp
                                frame.cpp
                                keyframe.cpp
                                feature.cpp
                                mappoint.cpp
                                frontend.cpp
                                backend.cpp
                                viewer.cpp
                                ORBextractor.cpp
                                camera.cpp
                                map.cpp
                            )



target_link_libraries(myslam ${THIRD_PARTY_LIBS})