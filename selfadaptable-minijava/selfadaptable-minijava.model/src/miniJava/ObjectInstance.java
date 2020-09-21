/**
 */
package miniJava;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Object Instance</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link miniJava.ObjectInstance#getFieldbindings <em>Fieldbindings</em>}</li>
 *   <li>{@link miniJava.ObjectInstance#getType <em>Type</em>}</li>
 * </ul>
 *
 * @see miniJava.MiniJavaPackage#getObjectInstance()
 * @model annotation="aspect"
 *        annotation="RuntimeData"
 * @generated
 */
public interface ObjectInstance extends EObject {
	/**
	 * Returns the value of the '<em><b>Fieldbindings</b></em>' containment reference list.
	 * The list contents are of type {@link miniJava.FieldBinding}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Fieldbindings</em>' containment reference list.
	 * @see miniJava.MiniJavaPackage#getObjectInstance_Fieldbindings()
	 * @model containment="true"
	 * @generated
	 */
	EList<FieldBinding> getFieldbindings();

	/**
	 * Returns the value of the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' reference.
	 * @see #setType(Clazz)
	 * @see miniJava.MiniJavaPackage#getObjectInstance_Type()
	 * @model required="true"
	 * @generated
	 */
	Clazz getType();

	/**
	 * Sets the value of the '{@link miniJava.ObjectInstance#getType <em>Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' reference.
	 * @see #getType()
	 * @generated
	 */
	void setType(Clazz value);

} // ObjectInstance
