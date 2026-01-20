#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cstdio>
#include <logger.h>
#include <macros.h>
#include <material.h>
#include <mesh.h>
#include <stdio.h>
#include <ogl.h>
#include <utility.h>
#include <filesystem>
#include <fstream>
#include <assimp/pbrmaterial.h>
#include <xatlas.h>
#if defined(DWSF_VULKAN)
#    include <vk_mem_alloc.h>
#endif

// Lightmap cache file format version
#define LIGHTMAP_CACHE_MAGIC 0x4C4D4150 // "LMAP"
#define LIGHTMAP_CACHE_VERSION 1

namespace dw
{
std::unordered_map<std::string, std::weak_ptr<Mesh>> Mesh::m_cache;

// Assimp texture enum lookup table.
static const aiTextureType kTextureTypes[] = {
    aiTextureType_DIFFUSE, aiTextureType_SPECULAR, aiTextureType_AMBIENT, aiTextureType_EMISSIVE, aiTextureType_HEIGHT, aiTextureType_NORMALS, aiTextureType_SHININESS, aiTextureType_OPACITY, aiTextureType_DISPLACEMENT, aiTextureType_LIGHTMAP, aiTextureType_REFLECTION
};

// Assimp texture enum string table.
static std::string kTextureTypeStrings[] = {
    "aiTextureType_DIFFUSE", "aiTextureType_SPECULAR", "aiTextureType_AMBIENT", "aiTextureType_EMISSIVE", "aiTextureType_HEIGHT", "aiTextureType_NORMALS", "aiTextureType_SHININESS", "aiTextureType_OPACITY", "aiTextureType_DISPLACEMENT", "aiTextureType_LIGHTMAP", "aiTextureType_REFLECTION"
};

std::string get_gltf_base_color_texture_path(aiMaterial* material)
{
    aiString path;
    aiReturn result = material->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_TEXTURE, &path);

    if (result == aiReturn_FAILURE)
        return "";
    else
    {
        std::string cppStr = std::string(path.C_Str());

        if (cppStr == "")
            return "";

        return cppStr;
    }
}

std::string get_gltf_metallic_roughness_texture_path(aiMaterial* material)
{
    aiString path;
    aiReturn result = material->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &path);

    if (result == aiReturn_FAILURE)
        return "";
    else
    {
        std::string cppStr = std::string(path.C_Str());

        if (cppStr == "")
            return "";

        return cppStr;
    }
}

std::string resolve_relative_path(const std::string& mesh_path, const std::string& path, bool is_gltf)
{
    if (is_gltf)
        return utility::path_without_file(mesh_path) + "/" + path;
    else
        return path;
}

static uint32_t g_last_mesh_idx = 0;

// -----------------------------------------------------------------------------------------------------------------------------------
// Assimp loader helper method declarations.
// -----------------------------------------------------------------------------------------------------------------------------------

std::string assimp_get_texture_path(aiMaterial*   material,
                                    aiTextureType texture_type);
bool        assimp_does_material_exist(std::vector<unsigned int>& materials,
                                       unsigned int&              current_material);

// -----------------------------------------------------------------------------------------------------------------------------------

Mesh::Ptr Mesh::load(
#if defined(DWSF_VULKAN)
    vk::Backend::Ptr backend,
#endif
    const std::string& path,
    bool               load_materials,
    bool               is_orca_mesh,
    bool               generate_lightmap_uv)
{
    std::filesystem::path absolute_file_path = std::filesystem::path(path);

    if (!absolute_file_path.is_absolute())
        absolute_file_path = std::filesystem::current_path() / absolute_file_path;
    DW_LOG_INFO("Load " + absolute_file_path.string());
    std::string absolute_file_path_str = absolute_file_path.string();

    // Cache key includes lightmap generation flag to differentiate cached meshes
    std::string cache_key = absolute_file_path_str + (generate_lightmap_uv ? "_lightmap" : "");

    if (m_cache.find(cache_key) == m_cache.end() || m_cache[cache_key].expired())
    {
        Mesh::Ptr mesh = std::shared_ptr<Mesh>(new Mesh(
#if defined(DWSF_VULKAN)
            backend,
#endif
            absolute_file_path_str,
            load_materials,
            is_orca_mesh,
            generate_lightmap_uv));
        m_cache[cache_key] = mesh;
        return mesh;
    }
    else
    {
        auto ptr = m_cache[cache_key];
        return ptr.lock();
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

Mesh::Ptr Mesh::load(
#if defined(DWSF_VULKAN)
    vk::Backend::Ptr backend,
#endif
    const std::string&                     name,
    std::vector<Vertex>                    vertices,
    std::vector<uint32_t>                  indices,
    std::vector<SubMesh>                   sub_meshes,
    std::vector<std::shared_ptr<Material>> materials,
    glm::vec3                              max_extents,
    glm::vec3                              min_extents)
{
    if (m_cache.find(name) == m_cache.end() || m_cache[name].expired())
    {
        Mesh::Ptr mesh = std::shared_ptr<Mesh>(new Mesh());

        // Manually assign properties...
        mesh->m_vertices    = vertices;
        mesh->m_materials   = materials;
        mesh->m_indices     = indices;
        mesh->m_sub_meshes  = sub_meshes;
        mesh->m_max_extents = max_extents;
        mesh->m_min_extents = min_extents;

        // ...then manually call the method to create GPU objects.
        mesh->create_gpu_objects(
#if defined(DWSF_VULKAN)
            backend
#endif
        );

        m_cache[name] = mesh;
        return mesh;
    }
    else
    {
        auto ptr = m_cache[name];
        return ptr.lock();
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

#if defined(DWSF_VULKAN)

void Mesh::initialize_for_ray_tracing(vk::Backend::Ptr backend)
{
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> build_ranges;
    std::vector<VkAccelerationStructureGeometryKHR>       geometries;
    std::vector<uint32_t>                                 max_primitive_counts;

    // Populate geometries
    for (int i = 0; i < m_sub_meshes.size(); i++)
    {
        Material::Ptr material = m_materials[m_sub_meshes[i].mat_idx];

        VkAccelerationStructureGeometryKHR geometry;
        DW_ZERO_MEMORY(geometry);

        VkGeometryFlagsKHR geometry_flags = 0;

        if (!material->alpha_test())
            geometry_flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        geometry.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.pNext                                       = nullptr;
        geometry.geometryType                                = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.pNext                    = nullptr;
        geometry.geometry.triangles.vertexData.deviceAddress = m_vbo->device_address();
        geometry.geometry.triangles.vertexStride             = sizeof(Vertex);
        geometry.geometry.triangles.maxVertex                = m_vertices.size() - 1;
        geometry.geometry.triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.indexData.deviceAddress  = m_ibo->device_address();
        geometry.geometry.triangles.indexType                = VK_INDEX_TYPE_UINT32;
        geometry.flags                                       = geometry_flags;

        geometries.push_back(geometry);
        max_primitive_counts.push_back(m_sub_meshes[i].index_count / 3);

        VkAccelerationStructureBuildRangeInfoKHR build_range;
        DW_ZERO_MEMORY(build_range);

        build_range.primitiveCount  = m_sub_meshes[i].index_count / 3;
        build_range.primitiveOffset = m_sub_meshes[i].base_index * sizeof(uint32_t);
        build_range.firstVertex     = 0;
        build_range.transformOffset = 0;

        build_ranges.push_back(build_range);
    }

    vk::BatchUploader uploader(backend);

    // Create blas
    vk::AccelerationStructure::Desc desc;

    desc.set_type(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    desc.set_flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    desc.set_geometries(geometries);
    desc.set_geometry_count(geometries.size());
    desc.set_max_primitive_counts(max_primitive_counts);

    m_blas = vk::AccelerationStructure::create(backend, desc);

    uploader.build_blas(m_blas, geometries, build_ranges);

    uploader.submit();
}

#endif

// -----------------------------------------------------------------------------------------------------------------------------------

bool Mesh::is_loaded(const std::string& name)
{
    return m_cache.find(name) != m_cache.end();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Mesh::load_from_disk(
#if defined(DWSF_VULKAN)
    vk::Backend::Ptr backend,
#endif
    const std::string& path,
    bool               load_materials,
    bool               is_orca_mesh)
{
    const aiScene*   Scene;
    Assimp::Importer importer;
    Scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    bool        is_gltf   = false;
    std::string extension = utility::file_extension(path);

    if (extension == "gltf" || extension == "glb")
        is_gltf = true;

    // Check if model has existing lightmap UV (second UV channel), log if found
    for (unsigned int i = 0; i < Scene->mNumMeshes; i++)
    {
        if (Scene->mMeshes[i]->HasTextureCoords(1))
        {
            DW_LOG_WARNING("Model has existing second UV set which will be replaced by xatlas-generated lightmap UV");
            break;
        }
    }

    m_sub_meshes.resize(Scene->mNumMeshes);

    // Temporary variables
    aiMaterial*                                 temp_material;
    std::vector<uint32_t>                       processed_mat_id;
    std::unordered_map<uint32_t, Material::Ptr> mat_id_mapping;
    std::unordered_map<uint32_t, uint32_t>      local_mat_idx_mapping;

    uint32_t vertex_count = 0;
    uint32_t index_count  = 0;

    // Iterate over submeshes and find materials
    for (int i = 0; i < m_sub_meshes.size(); i++)
    {
        bool has_least_one_texture = false;

        m_sub_meshes[i].name         = std::string(Scene->mMeshes[i]->mName.C_Str());
        m_sub_meshes[i].index_count  = Scene->mMeshes[i]->mNumFaces * 3;
        m_sub_meshes[i].base_index   = index_count;
        m_sub_meshes[i].base_vertex  = vertex_count;
        m_sub_meshes[i].vertex_count = Scene->mMeshes[i]->mNumVertices;

        vertex_count += Scene->mMeshes[i]->mNumVertices;
        index_count += m_sub_meshes[i].index_count;

        if (load_materials)
        {
            std::vector<std::string> texture_paths;

            int32_t    albedo_idx    = -1;
            int32_t    normal_idx    = -1;
            glm::ivec2 roughness_idx = glm::ivec2(-1);
            glm::ivec2 metallic_idx  = glm::ivec2(-1);
            int32_t    emissive_idx  = -1;

            glm::vec4 albedo_value    = glm::vec4(1.0f);
            float     roughness_value = 1.0f;
            float     metallic_value  = 0.0f;
            glm::vec3 emissive_value  = glm::vec3(0.0f);

            if (mat_id_mapping.find(Scene->mMeshes[i]->mMaterialIndex) == mat_id_mapping.end())
            {
                std::string current_mat_name;

                temp_material    = Scene->mMaterials[Scene->mMeshes[i]->mMaterialIndex];
                current_mat_name = path + std::to_string(i);

                aiColor3D diffuse         = aiColor3D(1.0f, 1.0f, 1.0f);
                bool      has_diifuse_val = false;

                // If this is a GLTF, try to find the base color texture path
                if (is_gltf)
                {
                    std::string texture_path = get_gltf_base_color_texture_path(temp_material);

                    if (!texture_path.empty())
                    {
                        albedo_idx = texture_paths.size();
                        texture_paths.push_back(resolve_relative_path(path, texture_path, is_gltf));
                    }
                }
                else
                {
                    // If not, try to find the Diffuse texture path
                    std::string texture_path = assimp_get_texture_path(temp_material, aiTextureType_DIFFUSE);

                    // If that doesn't exist, try to find Diffuse texture
                    if (texture_path.empty())
                        texture_path = assimp_get_texture_path(temp_material, aiTextureType_BASE_COLOR);

                    if (!texture_path.empty())
                    {
                        albedo_idx = texture_paths.size();
                        texture_paths.push_back(resolve_relative_path(path, texture_path, is_gltf));
                    }
                }

                if (albedo_idx == -1)
                {
                    aiColor3D diffuse = aiColor3D(1.0f, 1.0f, 1.0f);
                    float     alpha   = 1.0f;

                    // Try loading in a Diffuse material property
                    if (temp_material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) != AI_SUCCESS)
                        temp_material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR, diffuse);

                    temp_material->Get(AI_MATKEY_OPACITY, alpha);
#if defined(MATERIAL_LOG)
                    printf("Albedo Color: %f, %f, %f \n", diffuse.r, diffuse.g, diffuse.b);
#endif

                    albedo_value = glm::vec4(diffuse.r, diffuse.g, diffuse.b, alpha);
                }
                else
                {
#if defined(MATERIAL_LOG)
                    printf("Albedo Path: %s \n", albedo_path.c_str());
#endif
                    std::string texture_path = texture_paths[albedo_idx];

                    std::replace(texture_path.begin(), texture_path.end(), '\\', '/');

                    texture_paths[albedo_idx] = texture_path;
                }

                if (is_orca_mesh)
                {
                    std::string roughness_metallic_path = assimp_get_texture_path(temp_material, aiTextureType_SPECULAR);

                    if (!roughness_metallic_path.empty())
                    {
#if defined(MATERIAL_LOG)
                        printf("Roughness Metallic Path: %s \n", roughness_metallic_path.c_str());
#endif
                        std::replace(roughness_metallic_path.begin(), roughness_metallic_path.end(), '\\', '/');

                        roughness_idx.x = texture_paths.size();
                        roughness_idx.y = 1;

                        metallic_idx.x = texture_paths.size();
                        metallic_idx.y = 2;

                        texture_paths.push_back(resolve_relative_path(path, roughness_metallic_path, is_gltf));
                    }
                }
                else
                {
                    // Try to find Roughness texture
                    std::string roughness_path = assimp_get_texture_path(temp_material, aiTextureType_SHININESS);

                    if (roughness_path.empty())
                        roughness_path = get_gltf_metallic_roughness_texture_path(temp_material);

                    if (roughness_path.empty())
                    {
                        // Try loading in a Diffuse material property
                        temp_material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR, roughness_value);
#if defined(MATERIAL_LOG)
                        printf("Roughness Color: %f \n", roughness);
#endif
                    }
                    else
                    {
#if defined(MATERIAL_LOG)
                        printf("Roughness Path: %s \n", roughness_path.c_str());
#endif
                        std::replace(roughness_path.begin(), roughness_path.end(), '\\', '/');

                        roughness_idx.x = texture_paths.size();
                        roughness_idx.y = is_gltf ? 1 : 0;

                        texture_paths.push_back(resolve_relative_path(path, roughness_path, is_gltf));
                    }

                    // Try to find Metallic texture
                    std::string metallic_path = assimp_get_texture_path(temp_material, aiTextureType_AMBIENT);

                    if (metallic_path.empty())
                        metallic_path = get_gltf_metallic_roughness_texture_path(temp_material);

                    if (metallic_path.empty())
                    {
                        // Try loading in a Diffuse material property
                        temp_material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, metallic_value);
#if defined(MATERIAL_LOG)
                        printf("Metallic Color: %f \n", metallic);
#endif
                    }
                    else
                    {
#if defined(MATERIAL_LOG)
                        printf("Metallic Path: %s \n", metallic_path.c_str());
#endif
                        std::replace(metallic_path.begin(), metallic_path.end(), '\\', '/');

                        metallic_idx.x = texture_paths.size();
                        metallic_idx.y = is_gltf ? 2 : 0;

                        texture_paths.push_back(resolve_relative_path(path, metallic_path, is_gltf));
                    }
                }

                // Try to find Emissive texture
                std::string emissive_path = assimp_get_texture_path(temp_material, aiTextureType_EMISSIVE);

                if (emissive_path.empty())
                {
                    aiColor3D emissive;

                    // Try loading in a Emissive material property
                    if (temp_material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive))
                    {
#if defined(MATERIAL_LOG)
                        printf("Emissive Color: %f, %f, %f \n", emissive.r, emissive.g, emissive.b);
#endif
                        emissive_value.r = emissive.r;
                        emissive_value.g = emissive.g;
                        emissive_value.b = emissive.b;
                    }
                }
                else
                {
#if defined(MATERIAL_LOG)
                    printf("Emissive Path: %s \n", emissive_path.c_str());
#endif
                    std::replace(emissive_path.begin(), emissive_path.end(), '\\', '/');

                    emissive_idx = texture_paths.size();
                    texture_paths.push_back(resolve_relative_path(path, emissive_path, is_gltf));
                }

                // Try to find Normal texture
                std::string normal_path = assimp_get_texture_path(temp_material, aiTextureType_NORMALS);

                if (normal_path.empty())
                    normal_path = assimp_get_texture_path(temp_material, aiTextureType_HEIGHT);

                if (!normal_path.empty())
                {
#if defined(MATERIAL_LOG)
                    printf("Normal Path: %s \n", normal_path.c_str());
#endif
                    std::replace(normal_path.begin(), normal_path.end(), '\\', '/');

                    normal_idx = texture_paths.size();
                    texture_paths.push_back(resolve_relative_path(path, normal_path, is_gltf));
                }

                Material::Ptr mat = Material::load(
#if defined(DWSF_VULKAN)
                    backend,
#endif
                    texture_paths,
                    albedo_idx,
                    normal_idx,
                    roughness_idx,
                    metallic_idx,
                    emissive_idx);

                mat->set_albedo_value(albedo_value);
                mat->set_roughness_value(roughness_value);
                mat->set_metallic_value(metallic_value);
                mat->set_emissive_value(emissive_value);

                mat_id_mapping[Scene->mMeshes[i]->mMaterialIndex]        = mat;
                local_mat_idx_mapping[Scene->mMeshes[i]->mMaterialIndex] = m_materials.size();

                m_sub_meshes[i].mat_idx = m_materials.size();

                m_materials.push_back(mat);
            }
            else // if already exists, find the pointer.
                m_sub_meshes[i].mat_idx = local_mat_idx_mapping[Scene->mMeshes[i]->mMaterialIndex];
        }
    }

    m_vertices.resize(vertex_count);
    m_indices.resize(index_count);

    std::vector<uint32_t> temp_indices(index_count);

    aiMesh* temp_mesh;
    int     idx         = 0;
    int     vertexIndex = 0;

    // Iterate over submeshes...
    for (int i = 0; i < m_sub_meshes.size(); i++)
    {
        temp_mesh                   = Scene->mMeshes[i];
        m_sub_meshes[i].max_extents = glm::vec3(temp_mesh->mVertices[0].x, temp_mesh->mVertices[0].y, temp_mesh->mVertices[0].z);
        m_sub_meshes[i].min_extents = glm::vec3(temp_mesh->mVertices[0].x, temp_mesh->mVertices[0].y, temp_mesh->mVertices[0].z);

        uint32_t mat_id = 0;

        if (mat_id_mapping[Scene->mMeshes[i]->mMaterialIndex])
            mat_id = mat_id_mapping[Scene->mMeshes[i]->mMaterialIndex]->id();
        mat_id = m_sub_meshes[i].mat_idx;

        // Iterate over vertices in submesh...
        for (int k = 0; k < Scene->mMeshes[i]->mNumVertices; k++)
        {
            // Assign vertex values.
            m_vertices[vertexIndex].position = glm::vec4(temp_mesh->mVertices[k].x, temp_mesh->mVertices[k].y, temp_mesh->mVertices[k].z, float(mat_id));
            glm::vec3 n                      = glm::vec3(temp_mesh->mNormals[k].x, temp_mesh->mNormals[k].y, temp_mesh->mNormals[k].z);
            m_vertices[vertexIndex].normal   = glm::vec4(n, 0.0f);

            if (temp_mesh->mTangents && temp_mesh->mBitangents)
            {
                glm::vec3 t = glm::vec3(temp_mesh->mTangents[k].x, temp_mesh->mTangents[k].y, temp_mesh->mTangents[k].z);
                glm::vec3 b = glm::vec3(temp_mesh->mBitangents[k].x, temp_mesh->mBitangents[k].y, temp_mesh->mBitangents[k].z);

                // Assuming right handed coordinate space
                if (glm::dot(glm::cross(n, t), b) < 0.0f)
                    t *= -1.0f; // Flip tangent

                m_vertices[vertexIndex].tangent   = glm::vec4(t, 0.0f);
                m_vertices[vertexIndex].bitangent = glm::vec4(b, 0.0f);
            }

            // Find submesh bounding box extents.
            if (m_vertices[vertexIndex].position.x > m_sub_meshes[i].max_extents.x)
                m_sub_meshes[i].max_extents.x = m_vertices[vertexIndex].position.x;
            if (m_vertices[vertexIndex].position.y > m_sub_meshes[i].max_extents.y)
                m_sub_meshes[i].max_extents.y = m_vertices[vertexIndex].position.y;
            if (m_vertices[vertexIndex].position.z > m_sub_meshes[i].max_extents.z)
                m_sub_meshes[i].max_extents.z = m_vertices[vertexIndex].position.z;

            if (m_vertices[vertexIndex].position.x < m_sub_meshes[i].min_extents.x)
                m_sub_meshes[i].min_extents.x = m_vertices[vertexIndex].position.x;
            if (m_vertices[vertexIndex].position.y < m_sub_meshes[i].min_extents.y)
                m_sub_meshes[i].min_extents.y = m_vertices[vertexIndex].position.y;
            if (m_vertices[vertexIndex].position.z < m_sub_meshes[i].min_extents.z)
                m_sub_meshes[i].min_extents.z = m_vertices[vertexIndex].position.z;

            // Assign primary texture coordinates (uv0)
            if (temp_mesh->HasTextureCoords(0))
                m_vertices[vertexIndex].tex_coord = glm::vec4(temp_mesh->mTextureCoords[0][k].x, temp_mesh->mTextureCoords[0][k].y, 0.0f, 0.0f);

            // Lightmap UV (uv1): ignore any existing second UV set from model,
            // will be regenerated by xatlas for consistent quality
            m_vertices[vertexIndex].lightmap_tex_coord = glm::vec4(0.0f);

            vertexIndex++;
        }

        // Assign indices.
        for (int j = 0; j < temp_mesh->mNumFaces; j++)
        {
            temp_indices[idx] = temp_mesh->mFaces[j].mIndices[0];
            idx++;
            temp_indices[idx] = temp_mesh->mFaces[j].mIndices[1];
            idx++;
            temp_indices[idx] = temp_mesh->mFaces[j].mIndices[2];
            idx++;
        }
    }

    int count = 0;

    for (int i = 0; i < m_sub_meshes.size(); i++)
    {
        SubMesh& submesh = m_sub_meshes[i];

        for (int idx = submesh.base_index; idx < (submesh.base_index + submesh.index_count); idx++)
            m_indices[count++] = submesh.base_vertex + temp_indices[idx];

        submesh.base_vertex = 0;
    }

    m_max_extents = m_sub_meshes[0].max_extents;
    m_min_extents = m_sub_meshes[0].min_extents;

    // Find bounding box extents of entire mesh.
    for (int i = 0; i < m_sub_meshes.size(); i++)
    {
        if (m_sub_meshes[i].max_extents.x > m_max_extents.x)
            m_max_extents.x = m_sub_meshes[i].max_extents.x;
        if (m_sub_meshes[i].max_extents.y > m_max_extents.y)
            m_max_extents.y = m_sub_meshes[i].max_extents.y;
        if (m_sub_meshes[i].max_extents.z > m_max_extents.z)
            m_max_extents.z = m_sub_meshes[i].max_extents.z;

        if (m_sub_meshes[i].min_extents.x < m_min_extents.x)
            m_min_extents.x = m_sub_meshes[i].min_extents.x;
        if (m_sub_meshes[i].min_extents.y < m_min_extents.y)
            m_min_extents.y = m_sub_meshes[i].min_extents.y;
        if (m_sub_meshes[i].min_extents.z < m_min_extents.z)
            m_min_extents.z = m_sub_meshes[i].min_extents.z;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Mesh::create_gpu_objects(
#if defined(DWSF_VULKAN)
    vk::Backend::Ptr backend
#endif
)
{
#if defined(DWSF_VULKAN)
    m_vbo = vk::Buffer::create(backend, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, sizeof(Vertex) * m_vertices.size(), VMA_MEMORY_USAGE_GPU_ONLY, 0, &m_vertices[0]);
    m_ibo = vk::Buffer::create(backend, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, sizeof(uint32_t) * m_indices.size(), VMA_MEMORY_USAGE_GPU_ONLY, 0, &m_indices[0]);

    m_vertex_input_state_desc.add_binding_desc(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);

    m_vertex_input_state_desc.add_attribute_desc(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0);                                    // position
    m_vertex_input_state_desc.add_attribute_desc(1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tex_coord));          // uv0
    m_vertex_input_state_desc.add_attribute_desc(2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, lightmap_tex_coord)); // uv1 (lightmap)
    m_vertex_input_state_desc.add_attribute_desc(3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, normal));
    m_vertex_input_state_desc.add_attribute_desc(4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent));
    m_vertex_input_state_desc.add_attribute_desc(5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, bitangent));
#else
    // Create vertex buffer.
    m_vbo = gl::Buffer::create(GL_ARRAY_BUFFER, 0, sizeof(Vertex) * m_vertices.size(), m_vertices.data());

    if (!m_vbo)
        DW_LOG_ERROR("Failed to create Vertex Buffer");

    // Create index buffer.
    m_ibo = gl::Buffer::create(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(uint32_t) * m_indices.size(), m_indices.data());

    if (!m_ibo)
        DW_LOG_ERROR("Failed to create Index Buffer");

    // Declare vertex attributes.
    gl::VertexAttrib attribs[] = { { 4, GL_FLOAT, false, 0 },                                    // position
                                   { 4, GL_FLOAT, false, offsetof(Vertex, tex_coord) },          // uv0
                                   { 4, GL_FLOAT, false, offsetof(Vertex, lightmap_tex_coord) }, // uv1 (lightmap)
                                   { 4, GL_FLOAT, false, offsetof(Vertex, normal) },
                                   { 4, GL_FLOAT, false, offsetof(Vertex, tangent) },
                                   { 4, GL_FLOAT, false, offsetof(Vertex, bitangent) } };

    // Create vertex array.
    m_vao = gl::VertexArray::create(m_vbo, m_ibo, sizeof(Vertex), 6, attribs);

    if (!m_vao)
        DW_LOG_ERROR("Failed to create Vertex Array");
#endif
}

// -----------------------------------------------------------------------------------------------------------------------------------

Mesh::Mesh()
{
    m_id = g_last_mesh_idx++;
}

// -----------------------------------------------------------------------------------------------------------------------------------

Mesh::Mesh(
#if defined(DWSF_VULKAN)
    vk::Backend::Ptr backend,
#endif
    const std::string& path,
    bool               load_materials,
    bool               is_orca_mesh,
    bool               generate_lightmap_uv)
{
    m_id = g_last_mesh_idx++;

    load_from_disk(
#if defined(DWSF_VULKAN)
        backend,
#endif
        path,
        load_materials,
        is_orca_mesh);

    // Generate lightmap UV before creating GPU objects
    if (generate_lightmap_uv)
    {
        // Compute hash of original mesh geometry
        uint64_t    mesh_hash  = compute_mesh_hash();
        std::string cache_path = get_lightmap_cache_path(path);

        // Try to load from cache first
        if (!load_lightmap_cache(cache_path, mesh_hash))
        {
            // Cache miss or invalid, generate and save
            if (generate_lightmap_uvs())
            {
                save_lightmap_cache(cache_path, mesh_hash);
            }
        }
    }

    create_gpu_objects(
#if defined(DWSF_VULKAN)
        backend
#endif
    );
}

// -----------------------------------------------------------------------------------------------------------------------------------

Mesh::~Mesh()
{
    // Unload submesh materials.
    for (uint32_t i = 0; i < m_materials.size(); i++)
        m_materials[i].reset();

    m_ibo.reset();
    m_vbo.reset();
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Mesh::set_submesh_material(std::string name, std::shared_ptr<Material> material)
{
    for (int i = 0; i < m_sub_meshes.size(); i++)
    {
        if (name == m_sub_meshes[i].name)
        {
            m_sub_meshes[i].mat_idx = m_materials.size();
            m_materials.push_back(material);

            return true;
        }
    }

    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Mesh::set_submesh_material(uint32_t mesh_idx, std::shared_ptr<Material> material)
{
    if (mesh_idx >= m_sub_meshes.size())
        return false;

    m_sub_meshes[mesh_idx].mat_idx = m_materials.size();
    m_materials.push_back(material);

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Mesh::set_global_material(std::shared_ptr<Material> material)
{
    for (int i = 0; i < m_sub_meshes.size(); i++)
        m_sub_meshes[i].mat_idx = m_materials.size();

    m_materials.push_back(material);
}

// -----------------------------------------------------------------------------------------------------------------------------------
// Assimp loader helper method definitions
// -----------------------------------------------------------------------------------------------------------------------------------

std::string assimp_get_texture_path(aiMaterial*   material,
                                    aiTextureType texture_type)
{
    aiString path;
    aiReturn result = material->GetTexture(texture_type, 0, &path);

    if (result == aiReturn_FAILURE)
        return "";
    else
    {
        std::string cppStr = std::string(path.C_Str());

        if (cppStr == "")
            return "";

        return cppStr;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool assimp_does_material_exist(std::vector<unsigned int>& materials,
                                unsigned int&              current_material)
{
    for (auto it : materials)
    {
        if (it == current_material)
            return true;
    }

    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Mesh::generate_lightmap_uvs()
{
    if (m_vertices.empty() || m_indices.empty())
    {
        DW_LOG_ERROR("Cannot generate lightmap UVs: mesh has no geometry");
        return false;
    }

    xatlas::Atlas* atlas = xatlas::Create();

    // Add entire mesh as single xatlas mesh (all submeshes together)
    xatlas::MeshDecl mesh_decl;
    mesh_decl.vertexCount          = static_cast<uint32_t>(m_vertices.size());
    mesh_decl.vertexPositionData   = &m_vertices[0].position;
    mesh_decl.vertexPositionStride = sizeof(Vertex);
    mesh_decl.vertexNormalData     = &m_vertices[0].normal;
    mesh_decl.vertexNormalStride   = sizeof(Vertex);
    mesh_decl.vertexUvData         = &m_vertices[0].tex_coord;
    mesh_decl.vertexUvStride       = sizeof(Vertex);

    mesh_decl.indexCount  = static_cast<uint32_t>(m_indices.size());
    mesh_decl.indexData   = m_indices.data();
    mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl, 0);
    if (error != xatlas::AddMeshError::Success)
    {
        DW_LOG_ERROR(std::string("xatlas::AddMesh failed: ") + xatlas::StringForEnum(error));
        xatlas::Destroy(atlas);
        return false;
    }

    // Use xatlas default options (same as xatlas viewer)
    // ChartOptions defaults: normalDeviationWeight=2, roundnessWeight=0.01, straightnessWeight=6,
    //                        normalSeamWeight=4, textureSeamWeight=0.5, maxCost=2, maxIterations=1
    // PackOptions defaults:  bilinear=true, blockAlign=false, bruteForce=false, padding=0,
    //                        resolution=0, texelsPerUnit=0, rotateChartsToAxis=true
    xatlas::Generate(atlas);

    if (atlas->width == 0 || atlas->height == 0)
    {
        DW_LOG_ERROR("xatlas::Generate failed: atlas has zero dimensions");
        xatlas::Destroy(atlas);
        return false;
    }

    m_lightmap_width  = atlas->width;
    m_lightmap_height = atlas->height;

    // Build new vertex and index buffers with xatlas-generated UVs
    // xatlas may create new vertices due to UV seams
    // Since we added entire mesh as single xatlas mesh, we have exactly one output mesh
    if (atlas->meshCount != 1)
    {
        DW_LOG_ERROR("xatlas::Generate unexpected mesh count");
        xatlas::Destroy(atlas);
        return false;
    }

    const xatlas::Mesh& xmesh = atlas->meshes[0];

    std::vector<Vertex>   new_vertices;
    std::vector<uint32_t> new_indices;

    new_vertices.reserve(xmesh.vertexCount);
    new_indices.reserve(xmesh.indexCount);

    // Build mapping from old vertex index to new vertex indices
    // One old vertex may map to multiple new vertices due to UV seams
    std::vector<std::vector<uint32_t>> old_to_new_vertices(m_vertices.size());

    // Create new vertices with lightmap UVs
    for (uint32_t v = 0; v < xmesh.vertexCount; v++)
    {
        const xatlas::Vertex& xvert = xmesh.vertexArray[v];

        // xref points to the original vertex index
        Vertex new_vert = m_vertices[xvert.xref];

        // Set lightmap UV (normalized to 0-1 range)
        new_vert.lightmap_tex_coord.x = xvert.uv[0] / float(atlas->width);
        new_vert.lightmap_tex_coord.y = xvert.uv[1] / float(atlas->height);
        new_vert.lightmap_tex_coord.z = 0.0f;
        new_vert.lightmap_tex_coord.w = 0.0f;

        old_to_new_vertices[xvert.xref].push_back(static_cast<uint32_t>(new_vertices.size()));
        new_vertices.push_back(new_vert);
    }

    // Copy indices (already remapped by xatlas)
    for (uint32_t i = 0; i < xmesh.indexCount; i++)
        new_indices.push_back(xmesh.indexArray[i]);

    // Recalculate submesh boundaries based on new indices
    // Original submeshes had contiguous index ranges, we need to find new boundaries
    uint32_t current_index = 0;
    for (size_t submesh_idx = 0; submesh_idx < m_sub_meshes.size(); submesh_idx++)
    {
        SubMesh& submesh         = m_sub_meshes[submesh_idx];
        uint32_t old_base_index  = submesh.base_index;
        uint32_t old_index_count = submesh.index_count;

        // Submesh boundaries in new index buffer correspond to same triangle ranges
        // Since xatlas preserves triangle order, new submesh has same index count
        submesh.base_index   = current_index;
        submesh.base_vertex  = 0;                                          // All vertices in single buffer now
        submesh.vertex_count = static_cast<uint32_t>(new_vertices.size()); // Share all vertices

        current_index += old_index_count;
    }

    // Replace old data
    m_vertices         = std::move(new_vertices);
    m_indices          = std::move(new_indices);
    m_has_lightmap_uvs = true;

    xatlas::Destroy(atlas);

    DW_LOG_INFO(std::string("Generated lightmap UVs: ") + std::to_string(m_lightmap_width) + "x" + std::to_string(m_lightmap_height) + " atlas, " + std::to_string(m_vertices.size()) + " vertices, " + std::to_string(m_indices.size()) + " indices");

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string Mesh::get_lightmap_cache_path(const std::string& mesh_path)
{
    // Create cache file path: same directory as mesh, with .lightmap_cache extension
    const std::filesystem::path mesh_fs_path(mesh_path);
    const std::string           cache_name = mesh_fs_path.stem().string() + ".lightmap_cache";

    return (mesh_fs_path.parent_path() / cache_name).string();
}

// -----------------------------------------------------------------------------------------------------------------------------------

uint64_t Mesh::compute_mesh_hash() const
{
    // Simple FNV-1a hash of vertex positions and indices
    // This ensures cache invalidation when mesh geometry changes
    constexpr uint64_t FNV_PRIME  = 1099511628211ULL;
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;

    uint64_t hash = FNV_OFFSET;

    // Hash vertex positions
    for (const auto& v : m_vertices)
    {
        const auto* data = reinterpret_cast<const uint8_t*>(&v.position);
        for (size_t i = 0; i < sizeof(glm::vec4); i++)
        {
            hash ^= data[i];
            hash *= FNV_PRIME;
        }
    }

    // Hash indices
    for (const auto& idx : m_indices)
    {
        const auto* data = reinterpret_cast<const uint8_t*>(&idx);
        for (size_t i = 0; i < sizeof(uint32_t); i++)
        {
            hash ^= data[i];
            hash *= FNV_PRIME;
        }
    }

    // Hash submesh info
    for (const auto& sm : m_sub_meshes)
    {
        hash ^= sm.index_count;
        hash *= FNV_PRIME;
        hash ^= sm.base_index;
        hash *= FNV_PRIME;
    }

    return hash;
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Mesh::load_lightmap_cache(const std::string& cache_path, uint64_t expected_hash)
{
    std::ifstream file(cache_path, std::ios::binary);
    if (!file.is_open())
        return false;

    // Read and verify header
    uint32_t magic, version;
    uint64_t stored_hash;
    uint32_t vertex_count, index_count, submesh_count;
    uint32_t lightmap_width, lightmap_height;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&stored_hash), sizeof(stored_hash));
    file.read(reinterpret_cast<char*>(&vertex_count), sizeof(vertex_count));
    file.read(reinterpret_cast<char*>(&index_count), sizeof(index_count));
    file.read(reinterpret_cast<char*>(&submesh_count), sizeof(submesh_count));
    file.read(reinterpret_cast<char*>(&lightmap_width), sizeof(lightmap_width));
    file.read(reinterpret_cast<char*>(&lightmap_height), sizeof(lightmap_height));

    if (file.fail())
    {
        DW_LOG_WARNING("Lightmap cache read error: header");
        return false;
    }

    // Validate header
    if (magic != LIGHTMAP_CACHE_MAGIC)
    {
        DW_LOG_WARNING("Lightmap cache invalid magic");
        return false;
    }

    if (version != LIGHTMAP_CACHE_VERSION)
    {
        DW_LOG_WARNING("Lightmap cache version mismatch");
        return false;
    }

    if (stored_hash != expected_hash)
    {
        DW_LOG_WARNING("Lightmap cache hash mismatch - mesh geometry changed");
        return false;
    }

    // Read vertices
    std::vector<Vertex> vertices(vertex_count);
    file.read(reinterpret_cast<char*>(vertices.data()), static_cast<int32_t>(vertex_count) * sizeof(Vertex));

    // Read indices
    std::vector<uint32_t> indices(index_count);
    file.read(reinterpret_cast<char*>(indices.data()), static_cast<int32_t>(index_count) * sizeof(uint32_t));

    // Read submeshes
    std::vector<SubMesh> submeshes(submesh_count);
    for (uint32_t i = 0; i < submesh_count; i++)
    {
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        file.read(name.data(), name_len);
        submeshes[i].name = name;

        file.read(reinterpret_cast<char*>(&submeshes[i].mat_idx), sizeof(submeshes[i].mat_idx));
        file.read(reinterpret_cast<char*>(&submeshes[i].index_count), sizeof(submeshes[i].index_count));
        file.read(reinterpret_cast<char*>(&submeshes[i].base_vertex), sizeof(submeshes[i].base_vertex));
        file.read(reinterpret_cast<char*>(&submeshes[i].base_index), sizeof(submeshes[i].base_index));
        file.read(reinterpret_cast<char*>(&submeshes[i].vertex_count), sizeof(submeshes[i].vertex_count));
        file.read(reinterpret_cast<char*>(&submeshes[i].max_extents), sizeof(submeshes[i].max_extents));
        file.read(reinterpret_cast<char*>(&submeshes[i].min_extents), sizeof(submeshes[i].min_extents));
    }

    if (file.fail())
    {
        DW_LOG_WARNING("Lightmap cache read error: data");
        return false;
    }

    // Apply cached data
    m_vertices         = std::move(vertices);
    m_indices          = std::move(indices);
    m_sub_meshes       = std::move(submeshes);
    m_lightmap_width   = lightmap_width;
    m_lightmap_height  = lightmap_height;
    m_has_lightmap_uvs = true;

    DW_LOG_INFO(std::string("Loaded lightmap cache: ") + cache_path);
    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Mesh::save_lightmap_cache(const std::string& cache_path, const uint64_t mesh_hash) const
{
    std::ofstream file(cache_path, std::ios::binary);
    if (!file.is_open())
    {
        DW_LOG_WARNING(std::string("Failed to create lightmap cache: ") + cache_path);
        return false;
    }

    // Write header
    constexpr uint32_t magic         = LIGHTMAP_CACHE_MAGIC;
    constexpr uint32_t version       = LIGHTMAP_CACHE_VERSION;
    const uint32_t     vertex_count  = static_cast<uint32_t>(m_vertices.size());
    const uint32_t     index_count   = static_cast<uint32_t>(m_indices.size());
    const uint32_t     submesh_count = static_cast<uint32_t>(m_sub_meshes.size());

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&mesh_hash), sizeof(mesh_hash));
    file.write(reinterpret_cast<const char*>(&vertex_count), sizeof(vertex_count));
    file.write(reinterpret_cast<const char*>(&index_count), sizeof(index_count));
    file.write(reinterpret_cast<const char*>(&submesh_count), sizeof(submesh_count));
    file.write(reinterpret_cast<const char*>(&m_lightmap_width), sizeof(m_lightmap_width));
    file.write(reinterpret_cast<const char*>(&m_lightmap_height), sizeof(m_lightmap_height));

    // Write vertices
    file.write(reinterpret_cast<const char*>(m_vertices.data()), static_cast<int32_t>(vertex_count) * sizeof(Vertex));

    // Write indices
    file.write(reinterpret_cast<const char*>(m_indices.data()), static_cast<int32_t>(index_count) * sizeof(uint32_t));

    // Write submeshes
    for (const auto& sm : m_sub_meshes)
    {
        auto name_len = static_cast<uint32_t>(sm.name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(sm.name.data(), name_len);

        file.write(reinterpret_cast<const char*>(&sm.mat_idx), sizeof(sm.mat_idx));
        file.write(reinterpret_cast<const char*>(&sm.index_count), sizeof(sm.index_count));
        file.write(reinterpret_cast<const char*>(&sm.base_vertex), sizeof(sm.base_vertex));
        file.write(reinterpret_cast<const char*>(&sm.base_index), sizeof(sm.base_index));
        file.write(reinterpret_cast<const char*>(&sm.vertex_count), sizeof(sm.vertex_count));
        file.write(reinterpret_cast<const char*>(&sm.max_extents), sizeof(sm.max_extents));
        file.write(reinterpret_cast<const char*>(&sm.min_extents), sizeof(sm.min_extents));
    }

    if (file.fail())
    {
        DW_LOG_WARNING("Lightmap cache write error");
        return false;
    }

    DW_LOG_INFO(std::string("Saved lightmap cache: ") + cache_path);
    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace dw
