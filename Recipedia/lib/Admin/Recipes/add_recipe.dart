import 'package:flutter/material.dart';
import 'package:recipedia/Database/databaseHelper.dart';
import '../../WidgetsAndUtils/recipe_model.dart';
import '../admin_portal.dart';

class AddRecipe extends StatefulWidget {
  const AddRecipe({
    Key? key,
  }) : super(key: key);

  @override
  State<AddRecipe> createState() => _AddRecipeState();
}

class _AddRecipeState extends State<AddRecipe> {
  int recipeID = 0;
  String recipeName = "";
  TextEditingController recipeNameTextController = TextEditingController();
  String recipeCategory = "";
  TextEditingController recipeCategoryTextController = TextEditingController();
  String recipeDiet = "";
  TextEditingController recipeDietTextController = TextEditingController();
  String recipeCourse = "";
  TextEditingController recipeCourseTextController = TextEditingController();
  String recipeIngredients = '';
  TextEditingController recipeIngredientsTextController =
      TextEditingController();
  String recipeDescription = '';
  TextEditingController recipeDescriptionTextController =
      TextEditingController();
  String recipeTime = '';
  TextEditingController recipeTimeTextController = TextEditingController();
  String recipeURL = '';
  TextEditingController recipeURLTextController = TextEditingController();

  Future<void> getRecipeID() async {
    recipeID = await RecipeModel().getRecipeCount();
    setState(() {
      recipeID = recipeID + 100;
    });
  }

  @override
  void initState() {
    getRecipeID();
    super.initState();
  }

  Widget buildTextField(
      String labelText, TextEditingController textController, String variable) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            labelText,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
          ),
          TextField(
            enabled: true,
            decoration: const InputDecoration(
              filled: false,
              floatingLabelBehavior: FloatingLabelBehavior.always,
              fillColor: Colors.black45,
            ),
            style: const TextStyle(color: Colors.black, height: 0.75),
            autofocus: false,
            controller: textController,
            keyboardType: TextInputType.multiline,
            textInputAction: TextInputAction.done,
            onChanged: (text) {
              variable = text;
            },
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0XFFFF4F5A),
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Add Recipes',
          textAlign: TextAlign.center,
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.only(left: 15.0, right: 15.0, top: 30.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              //Recipe ID
              Row(
                children: [
                  const Text(
                    'Recipe ID: ',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    recipeID.toString(),
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.normal,
                    ),
                  ),
                ],
              ),
              SizedBox(
                height: size.height * 0.04,
              ),
              buildTextField(
                  'Recipe Name', recipeNameTextController, recipeName),
              buildTextField('Recipe Ingredients(Comma Seprated)',
                  recipeIngredientsTextController, recipeIngredients),
              buildTextField('Recipe Instruction',
                  recipeDescriptionTextController, recipeDescription),
              buildTextField(
                  'Recipe Diet', recipeDietTextController, recipeDiet),
              buildTextField(
                  'Recipe Course', recipeCourseTextController, recipeCourse),
              buildTextField('Recipe Category', recipeCategoryTextController,
                  recipeCategory),
              buildTextField(
                  'Recipe Time', recipeTimeTextController, recipeTime),
              buildTextField(
                  'Recipe Image URL', recipeURLTextController, recipeURL),
              Container(
                width: MediaQuery.of(context).size.width,
                height: 40,
                margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                decoration:
                    BoxDecoration(borderRadius: BorderRadius.circular(90)),
                child: ElevatedButton(
                  onPressed: () {
                    RecipeModel().addRecipe(
                      recipeID: recipeID,
                      recipeName: recipeName,
                      recipeDescription: recipeDescription,
                      recipeIngredients: recipeIngredients,
                      recipeImageURL: recipeURL,
                      recipeTime: recipeTime,
                      recipeDiet: recipeDiet,
                      recipeCourse: recipeCourse,
                      recipeRating: '5',
                      recipeCategory: recipeCategory,
                    );
                    DatabaseHelper().saveRecipe(
                      recipeID,
                      recipeName,
                      recipeDescription,
                      recipeCategory,
                      recipeDiet,
                      recipeCourse,
                      recipeIngredients,
                      recipeURL,
                      recipeTime,
                      5,
                    );
                    /*widget.recipes.add(Recipes(
                    recipeID: recipeID.toString(),
                    recipeName: recipeName,
                    recipeCategory: 'Category',
                    recipeDescription: recipeDescription,
                    recipeURL: recipeURL,
                    recipeRating: "0",
                    recipeTime: recipeTime,
                    recipeIngredients: recipeIngredients,
                  ));*/
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(seconds: 1),
                        transitionsBuilder:
                            (context, animation, animationTime, child) {
                          animation = CurvedAnimation(
                              parent: animation,
                              curve: Curves.fastLinearToSlowEaseIn);
                          return ScaleTransition(
                            scale: animation,
                            alignment: Alignment.center,
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return AdminPortal();
                        },
                      ),
                      (route) => false,
                    );
                  },
                  style: ButtonStyle(
                      backgroundColor:
                          MaterialStateProperty.resolveWith((states) {
                        if (states.contains(MaterialState.pressed)) {
                          return Colors.blueGrey;
                        }
                        return const Color(0XFFFF4F5A);
                      }),
                      shape: MaterialStateProperty.all(RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(30)))),
                  child: const Text(
                    'Add Recipe',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
