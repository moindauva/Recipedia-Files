import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:recipedia/Admin/admin_portal.dart';
import 'package:recipedia/Admin/recipes/recipes.dart';
import 'package:recipedia/Admin/recipes/update_recipe.dart';
import 'package:recipedia/Database/databaseHelper.dart';
import 'package:recipedia/WidgetsAndUtils/recipe_model.dart';
import 'package:recipedia/WidgetsAndUtils/toast_message.dart';
import '../../WidgetsAndUtils/progress_bar.dart';

class UpdateOrDeleteRecipe extends StatefulWidget {
  final String id;
  final String title;
  final String description;
  final int rating;
  final String cookTime;
  final String thumbnailUrl;
  final String diet;
  final String course;
  final String ingredients;
  final String category;

  const UpdateOrDeleteRecipe({
    Key? key,
    required this.id,
    required this.diet,
    required this.course,
    required this.title,
    required this.description,
    required this.cookTime,
    required this.rating,
    required this.thumbnailUrl,
    required this.ingredients,
    required this.category,
  }) : super(key: key);

  @override
  State<UpdateOrDeleteRecipe> createState() => _UpdateOrDeleteRecipeState();
}

class _UpdateOrDeleteRecipeState extends State<UpdateOrDeleteRecipe> {
  bool isEditEnabled = false;
  bool isEditIcon = true;
  Icon editIcon = const Icon(Icons.edit);
  String recipeID = '101';
  late List<String> descriptionInBulletPoints = widget.description.split('. ');
  late List<String> ingredientsInList = widget.ingredients.split(', ');

  final TextEditingController recipeNameTextController =
      TextEditingController();
  final TextEditingController recipeCourseTextController =
      TextEditingController();
  final TextEditingController recipeDietTextController =
      TextEditingController();
  final TextEditingController recipeDescriptionTextController =
      TextEditingController();
  final TextEditingController recipeRatingTextController =
      TextEditingController();
  final TextEditingController recipeTimeTextController =
      TextEditingController();
  final TextEditingController recipeIngredientsTextController =
      TextEditingController();

  BuildContext? dialogContext;
  List<Recipes> recipes = <Recipes>[];
  List<String> recipeIDs = [];
  List<String> recipeDescription = [];
  List<String> recipeName = [];
  List<String> recipeURL = [];
  List<String> recipeRating = [];
  List<String> recipeTime = [];
  List<String> recipeIngredients = [];

  Future<void> getAndDeleteRecipeData() async {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          dialogContext = context;
          return ProgressBar(
            message: "Loading..",
          );
        },
      );
    });
    //Deleting Process
    DatabaseHelper().deleteRecipe(int.parse(widget.id));
    RecipeModel().deleteRecipeData(widget.id);
    Navigator.pop(dialogContext!);
  }

  @override
  void initState() {
    recipeNameTextController.text = widget.title;
    recipeDescriptionTextController.text = widget.description;
    recipeRatingTextController.text = widget.rating.toString();
    recipeCourseTextController.text = widget.course;
    recipeDietTextController.text = widget.diet;
    recipeTimeTextController.text = widget.cookTime;
    recipeIngredientsTextController.text = widget.ingredients;

    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
        appBar: AppBar(
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.only(
                bottomRight: Radius.circular(25),
                bottomLeft: Radius.circular(25)),
          ),
          title: Text(
            widget.title,
          ),
          actions: <Widget>[
            IconButton(
              tooltip: 'Edit',
              onPressed: () {
                Navigator.push(
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
                      return UpdateRecipe(
                        recipeID: widget.id,
                        recipeCategory: widget.category,
                        recipeCourse: widget.course,
                        recipeDescription: widget.description,
                        recipeDiet: widget.diet,
                        recipeIngredients: widget.ingredients,
                        recipeName: widget.title,
                        recipeTime: widget.cookTime,
                        recipeURL: widget.thumbnailUrl, recipeRating: widget.rating,
                      );
                    },
                  ),
                );
                /*setState(() {
                  if (isEditIcon == true) {
                    isEditEnabled = true;
                    editIcon = const Icon(Icons.save);
                    isEditIcon = false;
                  } else {
                    editIcon = const Icon(Icons.edit);
                    isEditIcon = true;
                    isEditEnabled = false;
                  }
                });*/
              },
              icon: editIcon,
            ),
            IconButton(
              tooltip: 'Delete',
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (build) {
                    return AlertDialog(
                      title: Row(
                        children: [
                          Image.asset(
                            'assets/icon.png',
                            scale: 1.0,
                            height: 50,
                            width: 50,
                            fit: BoxFit.fitWidth,
                          ),
                          const Flexible(
                            child: Text(
                              '  Recipedia',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 24.0,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                      content: const Text(
                        'Are you sure, You want to delete the recipe?',
                        style: TextStyle(
                          fontWeight: FontWeight.normal,
                          fontSize: 16.0,
                        ),
                      ),
                      actions: [
                        TextButton(
                          child: const Text(
                            'Cancel',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16.0,
                            ),
                          ),
                          onPressed: () {
                            displayToastMessage('Cancel Pressed', context);
                          },
                        ),
                        TextButton(
                          child: const Text(
                            'Delete',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16.0,
                            ),
                          ),
                          onPressed: () async {
                            await getAndDeleteRecipeData();
                            // ignore: use_build_context_synchronously
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
                                pageBuilder:
                                    (context, animation, animationTime) {
                                  return AdminPortal();
                                },
                              ),
                              (route) => false,
                            );
                          },
                        )
                      ],
                    );
                  },
                );
              },
              icon: const Icon(
                Icons.delete,
              ),
            )
          ],
          backgroundColor: const Color(0XFFFF4F5A),
          elevation: 25.0,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            tooltip: 'Back',
            onPressed: () {
              Navigator.pop(context);
            },
          ),
          systemOverlayStyle: SystemUiOverlayStyle.light,
        ),
        body: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.only(left: 15.0, right: 15.0, top: 30.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                CircleAvatar(
                  backgroundImage: NetworkImage(widget.thumbnailUrl),
                  radius: 130.0,
                ),
                SizedBox(
                  height: size.height * 0.05,
                ),
                Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
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
                            widget.id,
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
                        'Recipe Name',
                        recipeNameTextController,
                        isEditEnabled,
                      ),
                      buildTextField(
                        'Recipe Ingredients',
                        recipeIngredientsTextController,
                        isEditEnabled,
                      ),
                      buildTextField(
                        'Recipe Description',
                        recipeDescriptionTextController,
                        isEditEnabled,
                      ),
                      buildTextField(
                        'Recipe Diet',
                        recipeDietTextController,
                        isEditEnabled,
                      ),
                      buildTextField(
                        'Recipe Course',
                        recipeCourseTextController,
                        isEditEnabled,
                      ),
                      buildTextField(
                        'Recipe Rating',
                        recipeRatingTextController,
                        isEditEnabled,
                      ),
                      buildTextField(
                        'Recipe Cooking Time',
                        recipeTimeTextController,
                        isEditEnabled,
                      ),
                    ]),
              ],
            ),
          ),
        ));
  }

  Widget buildTextField(String labelText, TextEditingController textController,
      bool isEditEnabled) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 15.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            labelText,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          TextField(
            maxLines: null,
            enabled: isEditEnabled,
            autocorrect: false,
            decoration: const InputDecoration(
              border: InputBorder.none,
              filled: false,
              floatingLabelBehavior: FloatingLabelBehavior.always,
              fillColor: Colors.black45,
            ),
            style: const TextStyle(
                color: Colors.black,
                fontWeight: FontWeight.normal,
                fontSize: 16,
                height: null),
            autofocus: false,
            controller: textController,
            keyboardType: TextInputType.emailAddress,
            textInputAction: TextInputAction.done,
          ),
        ],
      ),
    );
  }
}
