# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import cytomine
import sys

#connect to cytomine : parameters to set
cytomine_host="demo.cytomine.be"
cytomine_public_key="0ab78d51-3a6e-40e1-9b1d-d42c28bc1923"
cytomine_private_key="817d2e30-b4df-41d2-bb4b-fb29910b1d4e"
id_project=6575282

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#define software parameter template
software = conn.add_software("Landmark_Generic_Predictor", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("cytomine_models",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, True, 3 , False)

#add software to a given project
addSoftwareProject = conn.add_software_project(id_project,software.id)

print "Software id is %d"%software.id
