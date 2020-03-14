add_library(myslam SHARED 
                                config.cpp
                                system.cpp
                                frame.cpp
                                feature.cpp
                                frontend.cpp
                                viewer.cpp
                            )



target_link_libraries(myslam ${THIRD_PARTY_LIBS})