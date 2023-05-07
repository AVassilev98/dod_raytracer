file(GLOB assets 
    *.obj
    *.ply
    *.blend
)
foreach(asset ${assets})
  get_filename_component(fileName ${asset} NAME)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${fileName}
    ${CMAKE_CURRENT_BINARY_DIR}/${fileName}
    COPYONLY
  )
endforeach()
