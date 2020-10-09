package miniJava.interpreter.miniJava.impl;

import java.lang.Boolean;
import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.interpreter.miniJava.Block;
import miniJava.interpreter.miniJava.Clazz;
import miniJava.interpreter.miniJava.Member;
import miniJava.interpreter.miniJava.Method;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Parameter;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.TypeRef;
import minijava.MapService;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.EMap;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EStructuralFeature.Setting;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.EcoreEMap;
import org.eclipse.emf.ecore.util.InternalEList;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class MethodImpl extends MemberImpl implements Method {
	protected static final boolean ISABSTRACT_EDEFAULT = false;

	protected static final boolean ISSTATIC_EDEFAULT = false;

	protected boolean isabstract = ISABSTRACT_EDEFAULT;

	protected boolean isstatic = ISSTATIC_EDEFAULT;

	protected EList<Parameter> params;

	protected Block body;

	protected EMap<Clazz, Method> cache;

	protected MethodImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.METHOD;
	}

	public boolean isIsabstract() {
		return isabstract;
	}

	public void setIsabstract(boolean newIsabstract) {
		boolean oldIsabstract = isabstract;
		isabstract = newIsabstract;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD__ISABSTRACT, oldIsabstract, isabstract));
	}

	public boolean isIsstatic() {
		return isstatic;
	}

	public void setIsstatic(boolean newIsstatic) {
		boolean oldIsstatic = isstatic;
		isstatic = newIsstatic;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD__ISSTATIC, oldIsstatic, isstatic));
	}

	public EList<Parameter> getParams() {
		if (params == null) {
			params = new EObjectContainmentEList<Parameter>(Parameter.class, this, MiniJavaPackage.METHOD__PARAMS);
		}
		return params;
	}

	public Block getBody() {
		return body;
	}

	public NotificationChain basicSetBody(Block newBody, NotificationChain msgs) {
		Block oldBody = body;
		body = newBody;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD__BODY, oldBody, newBody);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setBody(Block newBody) {
		if (newBody != body) {
			NotificationChain msgs = null;
			if (body != null)
				msgs = ((InternalEObject) body).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.METHOD__BODY, null, msgs);
			if (newBody != null)
				msgs = ((InternalEObject) newBody).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.METHOD__BODY, null, msgs);
			msgs = basicSetBody(newBody, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD__BODY, newBody, newBody));
	}

	public EMap<Clazz, Method> getCache() {
		if (cache == null) {
			cache = new EcoreEMap<Clazz, Method>(MiniJavaPackage.Literals.CLAZZ_TO_METHOD_MAP, ClazzToMethodMapImpl.class, this, MiniJavaPackage.METHOD__CACHE);
		}
		return cache;
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.METHOD__PARAMS :
				return ((InternalEList<?>) getParams()).basicRemove(otherEnd, msgs);
			case MiniJavaPackage.METHOD__BODY :
				return basicSetBody(null, msgs);
			case MiniJavaPackage.METHOD__CACHE :
				return ((InternalEList<?>) getCache()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.METHOD__ISABSTRACT :
				return isIsabstract();
			case MiniJavaPackage.METHOD__ISSTATIC :
				return isIsstatic();
			case MiniJavaPackage.METHOD__PARAMS :
				return getParams();
			case MiniJavaPackage.METHOD__BODY :
				return getBody();
			case MiniJavaPackage.METHOD__CACHE :
				if (coreType) return getCache();
				else return getCache().map();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.METHOD__ISABSTRACT :
				setIsabstract((Boolean) newValue);
				return;
			case MiniJavaPackage.METHOD__ISSTATIC :
				setIsstatic((Boolean) newValue);
				return;
			case MiniJavaPackage.METHOD__PARAMS :
				getParams().clear();
				getParams().addAll((Collection<? extends Parameter>) newValue);
				return;
			case MiniJavaPackage.METHOD__BODY :
				setBody((Block) newValue);
				return;
			case MiniJavaPackage.METHOD__CACHE :
				((Setting)getCache()).set(newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD__ISABSTRACT :
				setIsabstract(ISABSTRACT_EDEFAULT);
				return;
			case MiniJavaPackage.METHOD__ISSTATIC :
				setIsstatic(ISSTATIC_EDEFAULT);
				return;
			case MiniJavaPackage.METHOD__PARAMS :
				getParams().clear();
				return;
			case MiniJavaPackage.METHOD__BODY :
				setBody((Block) null);
				return;
			case MiniJavaPackage.METHOD__CACHE :
				getCache().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD__ISABSTRACT :
				return isabstract != ISABSTRACT_EDEFAULT;
			case MiniJavaPackage.METHOD__ISSTATIC :
				return isstatic != ISSTATIC_EDEFAULT;
			case MiniJavaPackage.METHOD__PARAMS :
				return params != null && !params.isEmpty();
			case MiniJavaPackage.METHOD__BODY :
				return body != null;
			case MiniJavaPackage.METHOD__CACHE :
				return cache != null && !cache.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	public void evaluateStatement(State state) {
		((Block) (this.getBody())).evaluateStatement((State) (state));
	}

	public Method findOverride(Clazz c) {
		Method result;
		if (!(MapService.containsKey((EMap) (this.getCache()), (Clazz) (c)))) {
			Method that = ((Method) (this));
			if (CollectionService.exists(c.getMembers(), (x) -> EqualService.equals((x), (that)))) {
				result = (Method) (this) ;
			}
			else {
				int i = ((int) (0));
				Method found = ((Method) (null));
				while ((((i) < (CollectionService.size(c.getMembers()))) && (EqualService.equals((found), (null))))) {
					Member tmpm = ((Member) (CollectionService.get(c.getMembers(), i)));
					if (tmpm instanceof Method) {
						Method m = ((Method) (tmpm));
						if (((EqualService.equals((m.getName()), (this.name))) && (EqualService.equals((CollectionService.size(m.getParams())), (CollectionService.size(this.getParams())))))) {
							boolean compared = ((boolean) (((TypeRef) (m.getTypeRef())).compare((TypeRef) (this.typeRef))));
							int j = ((int) (0));
							int paramlgt = ((int) (CollectionService.size(m.getParams())));
							boolean alltrue = ((boolean) (compared));
							while ((((j) < (paramlgt)) && (alltrue))) {
								Parameter p1 = ((Parameter) (CollectionService.get(m.getParams(), j)));
								Parameter tmpp = ((Parameter) (CollectionService.head(CollectionService.select(this.getParams(), (p2) -> ((Parameter) (p1)).compare((Parameter) (p2))))));
								alltrue = !EqualService.equals((tmpp), (null));
								j = (j) + (1);
							}
							if (alltrue) {
								found = m;
							}
						}
					}
					i = (i) + (1);
				}
				if (!EqualService.equals((found), (null))) {
					result = (Method) (found) ;
				}
				else {
					if (!EqualService.equals((c.getSuperClass()), (null))) {
						result = (Method) (((Method) (this)).findOverride((Clazz) (c.getSuperClass()))) ;
					}
					else {
						result = (Method) (null) ;
					}
				}
			}
			MapService.put((EMap) (this.getCache()), (Clazz) (c), (Method) (result));
		}
		else {
			result = (Method) (MapService.getFromMap((EMap) (this.getCache()), (Clazz) (c))) ;
		}
		return result;
	}

	public void call(State state) {
		((Block) (this.getBody())).evaluateStatement((State) (state));
	}
}
