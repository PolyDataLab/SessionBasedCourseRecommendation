import os

#if the course is available in the user's previous baskets or course is not available in the offered list for that semester, we filer out this course
def filtering (item3, user_baskets, offered_course_list, item_dict):
    user_prev_items = []
    #Making a list of previous courses of given semester(s) for the user
    for basket1 in user_baskets:
        for item1 in basket1:
            if item1 not in user_prev_items:
                user_prev_items.append(item1)
    offered_course_list_new = []
    for item8 in offered_course_list:
        if item8 in item_dict:
            offered_course_list_new.append(item_dict[item8])
    
    if item3 in user_prev_items or item3 not in offered_course_list_new:
        return True
    else:
        return False

def create_folder(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass
